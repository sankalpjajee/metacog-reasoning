#!/usr/bin/env python3
"""
Generate training data for the confidence probe.

This script:
1. Loads samples from GSM8K, MMLU, and HellaSwag with stratified sampling
2. Runs self-consistency (N=3 samples) to measure agreement
3. Extracts hidden states from the LLM
4. Saves (hidden_state, confidence_label) pairs for probe training

Usage:
    python -m src.training.generate_probe_data \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --samples_per_benchmark 2000 \
        --output_dir data/training/probe_data
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import benchmark loaders
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.evaluation.benchmarks import (
    BenchmarkSample,
    GSM8kLoader,
    MMLULoader,
    HellaSwagLoader,
)


@dataclass
class ProbeTrainingSample:
    """A single training sample for the confidence probe."""
    sample_id: str
    benchmark: str
    category: str
    question: str
    ground_truth: str
    baseline_answers: List[str]
    agreement_rate: float
    confidence_label: int  # 0 = high confidence, 1 = low confidence
    hidden_state_path: str  # Path to saved hidden state tensor


class StratifiedSampler:
    """Stratified sampling across benchmark categories."""
    
    @staticmethod
    def sample_gsm8k(loader: GSM8kLoader, n_samples: int, split: str = "train") -> List[BenchmarkSample]:
        """Sample from GSM8K (single category, random sampling)."""
        samples = loader.load(split=split)
        if len(samples) <= n_samples:
            return samples
        return random.sample(samples, n_samples)
    
    @staticmethod
    def sample_mmlu(loader: MMLULoader, n_samples: int, split: str = "test") -> List[BenchmarkSample]:
        """Stratified sampling from MMLU across all subjects."""
        all_samples = loader.load(split=split)
        
        # Group by subject/category
        by_category = defaultdict(list)
        for sample in all_samples:
            by_category[sample.category].append(sample)
        
        n_categories = len(by_category)
        samples_per_category = max(1, n_samples // n_categories)
        
        print(f"MMLU: {n_categories} subjects, ~{samples_per_category} samples each")
        
        selected = []
        for category, category_samples in by_category.items():
            n_to_sample = min(samples_per_category, len(category_samples))
            selected.extend(random.sample(category_samples, n_to_sample))
        
        # If we need more samples, randomly add from remaining
        if len(selected) < n_samples:
            remaining = [s for s in all_samples if s not in selected]
            n_extra = min(n_samples - len(selected), len(remaining))
            selected.extend(random.sample(remaining, n_extra))
        
        # If we have too many, randomly remove
        if len(selected) > n_samples:
            selected = random.sample(selected, n_samples)
        
        return selected
    
    @staticmethod
    def sample_hellaswag(loader: HellaSwagLoader, n_samples: int, split: str = "validation") -> List[BenchmarkSample]:
        """Stratified sampling from HellaSwag across activity categories."""
        all_samples = loader.load(split=split)
        
        # Group by activity category
        by_category = defaultdict(list)
        for sample in all_samples:
            by_category[sample.category].append(sample)
        
        n_categories = len(by_category)
        samples_per_category = max(1, n_samples // n_categories)
        
        print(f"HellaSwag: {n_categories} activity categories, ~{samples_per_category} samples each")
        
        selected = []
        for category, category_samples in by_category.items():
            n_to_sample = min(samples_per_category, len(category_samples))
            selected.extend(random.sample(category_samples, n_to_sample))
        
        # Adjust to exact count
        if len(selected) < n_samples:
            remaining = [s for s in all_samples if s not in selected]
            n_extra = min(n_samples - len(selected), len(remaining))
            selected.extend(random.sample(remaining, n_extra))
        
        if len(selected) > n_samples:
            selected = random.sample(selected, n_samples)
        
        return selected


class SelfConsistencyGenerator:
    """Generate self-consistency samples and extract hidden states."""
    
    def __init__(
        self,
        model_path: str,
        n_samples: int = 3,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        device: str = "cuda"
    ):
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device
        
        print(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_baseline_prompt(self, question: str, benchmark: str) -> str:
        """Get the baseline prompt for a question."""
        if benchmark == "gsm8k":
            return f"""Solve this math problem step by step.

Question: {question}

Think through this carefully, then provide your final answer.
Final Answer: [just the number]"""
        
        elif benchmark == "mmlu":
            return f"""{question}

Think through this carefully, then provide your final answer.
Final Answer: [just the letter A, B, C, or D]"""
        
        elif benchmark == "hellaswag":
            return f"""{question}

Think through this carefully, then provide your final answer.
Final Answer: [just the letter A, B, C, or D]"""
        
        else:
            return f"""{question}

Final Answer:"""
    
    def format_chat_prompt(self, prompt: str) -> str:
        """Format prompt using chat template."""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate_samples(self, question: str, benchmark: str) -> List[str]:
        """Generate N samples for self-consistency."""
        prompt = self.get_baseline_prompt(question, benchmark)
        formatted_prompt = self.format_chat_prompt(prompt)
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        answers = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = self.extract_answer(response, benchmark)
            answers.append(answer)
        
        return answers
    
    def extract_answer(self, response: str, benchmark: str) -> str:
        """Extract the final answer from model response."""
        import re
        
        # Look for "Final Answer:" pattern
        patterns = [
            r'Final Answer:\s*([^\n]+)',
            r'final answer:\s*([^\n]+)',
            r'Answer:\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up the answer
                answer = re.sub(r'[^\w\d\.\-]', '', answer)
                if answer:
                    return answer
        
        # Fallback: look for specific patterns based on benchmark
        if benchmark == "gsm8k":
            # Look for numbers
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                return numbers[-1]
        
        elif benchmark in ["mmlu", "hellaswag"]:
            # Look for single letters A-D
            letters = re.findall(r'\b([A-D])\b', response)
            if letters:
                return letters[-1]
        
        return ""
    
    def extract_hidden_state(self, question: str, benchmark: str) -> torch.Tensor:
        """Extract hidden state from the model for a question."""
        prompt = self.get_baseline_prompt(question, benchmark)
        formatted_prompt = self.format_chat_prompt(prompt)
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )
            # Get last layer hidden state at last token position
            hidden_state = outputs.hidden_states[-1][0, -1, :].cpu()
        
        return hidden_state
    
    def calculate_agreement(self, answers: List[str]) -> float:
        """Calculate agreement rate among answers."""
        if not answers or all(a == "" for a in answers):
            return 0.0
        
        # Normalize answers
        normalized = [a.strip().upper() for a in answers if a.strip()]
        if not normalized:
            return 0.0
        
        # Count most common answer
        from collections import Counter
        counter = Counter(normalized)
        most_common_count = counter.most_common(1)[0][1]
        
        return most_common_count / len(normalized)


def generate_training_data(
    model_path: str,
    samples_per_benchmark: int,
    output_dir: str,
    agreement_threshold: float = 1.0,
    seed: int = 42
):
    """Generate training data for the confidence probe."""
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    hidden_states_dir = os.path.join(output_dir, "hidden_states")
    os.makedirs(hidden_states_dir, exist_ok=True)
    
    # Initialize loaders
    gsm8k_loader = GSM8kLoader()
    mmlu_loader = MMLULoader()
    hellaswag_loader = HellaSwagLoader()
    
    # Sample from each benchmark with stratified sampling
    print("\n" + "="*60)
    print("SAMPLING FROM BENCHMARKS")
    print("="*60)
    
    print(f"\nSampling {samples_per_benchmark} from GSM8K (train split)...")
    gsm8k_samples = StratifiedSampler.sample_gsm8k(gsm8k_loader, samples_per_benchmark, split="train")
    print(f"  Got {len(gsm8k_samples)} samples")
    
    print(f"\nSampling {samples_per_benchmark} from MMLU (test split, stratified by subject)...")
    mmlu_samples = StratifiedSampler.sample_mmlu(mmlu_loader, samples_per_benchmark, split="test")
    print(f"  Got {len(mmlu_samples)} samples")
    
    print(f"\nSampling {samples_per_benchmark} from HellaSwag (validation split, stratified by activity)...")
    hellaswag_samples = StratifiedSampler.sample_hellaswag(hellaswag_loader, samples_per_benchmark, split="validation")
    print(f"  Got {len(hellaswag_samples)} samples")
    
    # Combine all samples
    all_samples = [
        (s, "gsm8k") for s in gsm8k_samples
    ] + [
        (s, "mmlu") for s in mmlu_samples
    ] + [
        (s, "hellaswag") for s in hellaswag_samples
    ]
    
    print(f"\nTotal samples: {len(all_samples)}")
    
    # Initialize generator
    generator = SelfConsistencyGenerator(model_path=model_path)
    
    # Generate training data
    print("\n" + "="*60)
    print("GENERATING SELF-CONSISTENCY DATA")
    print("="*60)
    
    training_samples = []
    hidden_states = []
    
    for idx, (sample, benchmark) in enumerate(tqdm(all_samples, desc="Processing samples")):
        try:
            # Generate self-consistency samples
            answers = generator.generate_samples(sample.question, benchmark)
            
            # Calculate agreement rate
            agreement_rate = generator.calculate_agreement(answers)
            
            # Determine confidence label
            # 0 = high confidence (100% agreement)
            # 1 = low confidence (< 100% agreement)
            confidence_label = 0 if agreement_rate >= agreement_threshold else 1
            
            # Extract hidden state
            hidden_state = generator.extract_hidden_state(sample.question, benchmark)
            
            # Save hidden state
            hs_filename = f"hs_{idx:06d}.pt"
            hs_path = os.path.join(hidden_states_dir, hs_filename)
            torch.save(hidden_state, hs_path)
            
            # Create training sample
            training_sample = ProbeTrainingSample(
                sample_id=sample.id,
                benchmark=benchmark,
                category=sample.category or benchmark,
                question=sample.question[:500],  # Truncate for storage
                ground_truth=sample.answer,
                baseline_answers=answers,
                agreement_rate=agreement_rate,
                confidence_label=confidence_label,
                hidden_state_path=hs_filename,
            )
            
            training_samples.append(training_sample)
            hidden_states.append(hidden_state)
            
            # Progress update every 100 samples
            if (idx + 1) % 100 == 0:
                n_high = sum(1 for s in training_samples if s.confidence_label == 0)
                n_low = sum(1 for s in training_samples if s.confidence_label == 1)
                print(f"\n  Progress: {idx+1}/{len(all_samples)}")
                print(f"  High confidence: {n_high} ({100*n_high/(idx+1):.1f}%)")
                print(f"  Low confidence: {n_low} ({100*n_low/(idx+1):.1f}%)")
        
        except Exception as e:
            print(f"\n  Error processing sample {idx}: {e}")
            continue
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump([asdict(s) for s in training_samples], f, indent=2)
    
    # Save stacked hidden states tensor for efficient loading
    all_hidden_states = torch.stack(hidden_states)
    all_labels = torch.tensor([s.confidence_label for s in training_samples])
    
    tensors_path = os.path.join(output_dir, "training_tensors.pt")
    torch.save({
        'hidden_states': all_hidden_states,
        'labels': all_labels,
    }, tensors_path)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    n_high = sum(1 for s in training_samples if s.confidence_label == 0)
    n_low = sum(1 for s in training_samples if s.confidence_label == 1)
    
    print(f"\nTotal samples: {len(training_samples)}")
    print(f"High confidence (label=0): {n_high} ({100*n_high/len(training_samples):.1f}%)")
    print(f"Low confidence (label=1): {n_low} ({100*n_low/len(training_samples):.1f}%)")
    
    # Breakdown by benchmark
    print("\nBy benchmark:")
    for benchmark in ["gsm8k", "mmlu", "hellaswag"]:
        bench_samples = [s for s in training_samples if s.benchmark == benchmark]
        n_bench_high = sum(1 for s in bench_samples if s.confidence_label == 0)
        n_bench_low = sum(1 for s in bench_samples if s.confidence_label == 1)
        print(f"  {benchmark}: {len(bench_samples)} total, {n_bench_high} high, {n_bench_low} low")
    
    print(f"\nSaved to: {output_dir}")
    print(f"  - training_metadata.json: Sample metadata")
    print(f"  - training_tensors.pt: Stacked hidden states and labels")
    print(f"  - hidden_states/: Individual hidden state tensors")


def main():
    parser = argparse.ArgumentParser(description="Generate training data for confidence probe")
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the model"
    )
    parser.add_argument(
        "--samples_per_benchmark",
        type=int,
        default=2000,
        help="Number of samples per benchmark"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/training/probe_data",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--agreement_threshold",
        type=float,
        default=1.0,
        help="Agreement threshold for high confidence (default: 1.0 = 100%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    generate_training_data(
        model_path=args.model_path,
        samples_per_benchmark=args.samples_per_benchmark,
        output_dir=args.output_dir,
        agreement_threshold=args.agreement_threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
