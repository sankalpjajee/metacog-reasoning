#!/usr/bin/env python3
"""
Learned Adaptive Evaluator - Uses trained confidence probe for strategy selection.

This evaluator:
1. For each question, extracts hidden state from the LLM
2. Uses the trained probe to predict confidence (high/low)
3. If high confidence: uses baseline prompting
4. If low confidence: uses metacognitive prompting
5. Reports accuracy and method distribution

Usage:
    python -m src.evaluation.learned_adaptive_evaluator \
        --probe_path models/confidence_probe/best_probe.pt \
        --benchmarks gsm8k mmlu hellaswag \
        --num_samples 1000 \
        --output_dir results/learned_adaptive
"""

import argparse
import json
import os
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import benchmark loaders and probe
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.evaluation.benchmarks import (
    BenchmarkSample,
    GSM8kLoader,
    MMLULoader,
    HellaSwagLoader,
    load_benchmark,
)
from src.training.train_probe import ConfidenceProbe


@dataclass
class EvaluationResult:
    """Result for a single evaluation sample."""
    sample_id: str
    benchmark: str
    category: str
    question: str
    ground_truth: str
    predicted_answer: str
    is_correct: bool
    method_used: str  # "baseline" or "metacognitive"
    confidence_score: float
    raw_response: str


class LearnedAdaptiveEvaluator:
    """Evaluator that uses a trained probe for adaptive strategy selection."""
    
    def __init__(
        self,
        model_path: str,
        probe_path: str,
        confidence_threshold: float = 0.5,
        max_new_tokens: int = 512,
        device: str = "cuda"
    ):
        self.confidence_threshold = confidence_threshold
        self.max_new_tokens = max_new_tokens
        self.device = device
        
        # Load LLM
        print(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load probe
        print(f"Loading probe: {probe_path}")
        checkpoint = torch.load(probe_path, map_location=device)
        config = checkpoint['config']
        
        self.probe = ConfidenceProbe(
            hidden_size=config['hidden_size'],
            intermediate_size=config['intermediate_size'],
            dropout=config['dropout']
        ).to(device)
        self.probe.load_state_dict(checkpoint['model_state_dict'])
        self.probe.eval()
        
        print(f"Probe loaded. Val metrics: {checkpoint.get('val_metrics', 'N/A')}")
    
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
    
    def get_metacognitive_prompt(self, question: str, benchmark: str) -> str:
        """Get the metacognitive prompt for a question."""
        if benchmark == "gsm8k":
            return f"""You are solving a math problem. Before answering, engage in metacognitive reasoning:

1. UNDERSTAND: What is being asked? What information is given?
2. PLAN: What steps are needed to solve this?
3. MONITOR: As you work, check each step. Does it make sense?
4. VERIFY: Before finalizing, verify your answer is reasonable.

Question: {question}

Work through this problem with careful metacognitive monitoring:

UNDERSTANDING:
[Restate the problem in your own words]

PLANNING:
[Outline your solution approach]

SOLVING WITH MONITORING:
[Show your work, checking each step]

VERIFICATION:
[Verify your answer makes sense]

Final Answer: [just the number]"""
        
        elif benchmark == "mmlu":
            return f"""You are answering a knowledge question. Before answering, engage in metacognitive reasoning:

1. UNDERSTAND: What exactly is being asked?
2. RECALL: What relevant knowledge do I have?
3. EVALUATE: Consider each option carefully.
4. VERIFY: Am I confident in my choice? Why?

{question}

Work through this with careful metacognitive monitoring:

UNDERSTANDING:
[What is this question really asking?]

KNOWLEDGE RECALL:
[What do I know about this topic?]

OPTION ANALYSIS:
[Evaluate each option]

VERIFICATION:
[Why am I confident in my answer?]

Final Answer: [just the letter A, B, C, or D]"""
        
        elif benchmark == "hellaswag":
            return f"""You are completing a sentence about a common situation. Before answering, engage in metacognitive reasoning:

1. UNDERSTAND: What is the context? What's happening?
2. PREDICT: What would logically happen next?
3. EVALUATE: Which option best fits the context?
4. VERIFY: Does my choice make sense?

{question}

Work through this with careful metacognitive monitoring:

UNDERSTANDING:
[What is the situation described?]

PREDICTION:
[What would naturally happen next?]

OPTION ANALYSIS:
[Evaluate each option for fit]

VERIFICATION:
[Why does my choice make sense?]

Final Answer: [just the letter A, B, C, or D]"""
        
        else:
            return self.get_baseline_prompt(question, benchmark)
    
    def format_chat_prompt(self, prompt: str) -> str:
        """Format prompt using chat template."""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
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
            hidden_state = outputs.hidden_states[-1][0, -1, :]
        
        return hidden_state
    
    def predict_confidence(self, hidden_state: torch.Tensor) -> Tuple[str, float]:
        """
        Use the probe to predict confidence level.
        
        Returns:
            method: "baseline" if high confidence, "metacognitive" if low confidence
            score: confidence score (probability of low confidence)
        """
        with torch.no_grad():
            logits = self.probe(hidden_state.unsqueeze(0))
            prob = torch.sigmoid(logits).item()
        
        # prob is probability of LOW confidence (label=1)
        # If prob > threshold, use metacognitive (low confidence)
        # If prob <= threshold, use baseline (high confidence)
        if prob > self.confidence_threshold:
            return "metacognitive", prob
        else:
            return "baseline", prob
    
    def generate_response(self, question: str, benchmark: str, method: str) -> str:
        """Generate response using the specified method."""
        if method == "baseline":
            prompt = self.get_baseline_prompt(question, benchmark)
        else:
            prompt = self.get_metacognitive_prompt(question, benchmark)
        
        formatted_prompt = self.format_chat_prompt(prompt)
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,  # Greedy decoding for evaluation
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def extract_answer(self, response: str, benchmark: str) -> str:
        """Extract the final answer from model response."""
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
    
    def check_answer(self, predicted: str, ground_truth: str, benchmark: str) -> bool:
        """Check if the predicted answer is correct."""
        pred_norm = predicted.strip().upper()
        gt_norm = ground_truth.strip().upper()
        
        if benchmark == "gsm8k":
            # Normalize numbers
            try:
                pred_num = float(pred_norm.replace(',', ''))
                gt_num = float(gt_norm.replace(',', ''))
                return abs(pred_num - gt_num) < 1e-6
            except:
                return pred_norm == gt_norm
        else:
            # Multiple choice - compare letters
            return pred_norm == gt_norm
    
    def evaluate_sample(self, sample: BenchmarkSample, benchmark: str) -> EvaluationResult:
        """Evaluate a single sample."""
        # Extract hidden state
        hidden_state = self.extract_hidden_state(sample.question, benchmark)
        
        # Predict confidence and choose method
        method, confidence_score = self.predict_confidence(hidden_state)
        
        # Generate response
        response = self.generate_response(sample.question, benchmark, method)
        
        # Extract and check answer
        predicted_answer = self.extract_answer(response, benchmark)
        is_correct = self.check_answer(predicted_answer, sample.answer, benchmark)
        
        return EvaluationResult(
            sample_id=sample.id,
            benchmark=benchmark,
            category=sample.category or benchmark,
            question=sample.question[:500],
            ground_truth=sample.answer,
            predicted_answer=predicted_answer,
            is_correct=is_correct,
            method_used=method,
            confidence_score=confidence_score,
            raw_response=response[-1000:],  # Last 1000 chars
        )
    
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        num_samples: int = 1000
    ) -> Dict:
        """Evaluate on a benchmark."""
        print(f"\n{'='*60}")
        print(f"EVALUATING: {benchmark_name.upper()}")
        print(f"{'='*60}")
        
        # Load benchmark
        split = "validation" if benchmark_name == "hellaswag" else "test"
        samples = load_benchmark(benchmark_name, split=split)
        samples = samples[:num_samples]
        
        print(f"Loaded {len(samples)} samples")
        
        results = []
        for sample in tqdm(samples, desc=f"Evaluating {benchmark_name}"):
            try:
                result = self.evaluate_sample(sample, benchmark_name)
                results.append(result)
            except Exception as e:
                print(f"\nError on sample {sample.id}: {e}")
                continue
        
        # Calculate metrics
        n_correct = sum(1 for r in results if r.is_correct)
        n_total = len(results)
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        
        n_baseline = sum(1 for r in results if r.method_used == "baseline")
        n_metacog = sum(1 for r in results if r.method_used == "metacognitive")
        
        baseline_correct = sum(1 for r in results if r.method_used == "baseline" and r.is_correct)
        metacog_correct = sum(1 for r in results if r.method_used == "metacognitive" and r.is_correct)
        
        baseline_acc = baseline_correct / n_baseline if n_baseline > 0 else 0.0
        metacog_acc = metacog_correct / n_metacog if n_metacog > 0 else 0.0
        
        metrics = {
            'benchmark': benchmark_name,
            'num_samples': n_total,
            'accuracy': accuracy,
            'correct': n_correct,
            'method_distribution': {
                'baseline': n_baseline,
                'metacognitive': n_metacog,
            },
            'accuracy_by_method': {
                'baseline': baseline_acc,
                'metacognitive': metacog_acc,
            },
            'avg_confidence_score': sum(r.confidence_score for r in results) / n_total if n_total > 0 else 0.0,
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"RESULTS: {benchmark_name.upper()}")
        print(f"{'='*60}")
        print(f"Overall accuracy: {accuracy*100:.2f}% ({n_correct}/{n_total})")
        print(f"\nMethod distribution:")
        print(f"  Baseline: {n_baseline} ({100*n_baseline/n_total:.1f}%)")
        print(f"  Metacognitive: {n_metacog} ({100*n_metacog/n_total:.1f}%)")
        print(f"\nAccuracy by method:")
        print(f"  Baseline: {baseline_acc*100:.2f}% ({baseline_correct}/{n_baseline})")
        print(f"  Metacognitive: {metacog_acc*100:.2f}% ({metacog_correct}/{n_metacog})")
        print(f"\nAvg confidence score: {metrics['avg_confidence_score']:.3f}")
        
        return {
            'metrics': metrics,
            'results': [asdict(r) for r in results],
        }


def main():
    parser = argparse.ArgumentParser(description="Learned Adaptive Evaluator")
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the model"
    )
    parser.add_argument(
        "--probe_path",
        type=str,
        default="models/confidence_probe/best_probe.pt",
        help="Path to trained probe checkpoint"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gsm8k", "mmlu", "hellaswag"],
        help="Benchmarks to evaluate"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples per benchmark"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Threshold for confidence (above = use metacognitive)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/learned_adaptive",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = LearnedAdaptiveEvaluator(
        model_path=args.model_path,
        probe_path=args.probe_path,
        confidence_threshold=args.confidence_threshold,
    )
    
    # Evaluate each benchmark
    all_results = {}
    for benchmark in args.benchmarks:
        result = evaluator.evaluate_benchmark(benchmark, args.num_samples)
        all_results[benchmark] = result
        
        # Save individual benchmark results
        output_path = os.path.join(
            args.output_dir,
            f"learned_adaptive_{benchmark}_results.json"
        )
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    # Save combined summary
    summary = {
        'config': {
            'model_path': args.model_path,
            'probe_path': args.probe_path,
            'confidence_threshold': args.confidence_threshold,
            'num_samples': args.num_samples,
        },
        'results': {b: r['metrics'] for b, r in all_results.items()}
    }
    
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for benchmark, result in all_results.items():
        metrics = result['metrics']
        print(f"\n{benchmark.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Baseline used: {metrics['method_distribution']['baseline']} "
              f"({metrics['accuracy_by_method']['baseline']*100:.1f}% acc)")
        print(f"  Metacog used: {metrics['method_distribution']['metacognitive']} "
              f"({metrics['accuracy_by_method']['metacognitive']*100:.1f}% acc)")


if __name__ == "__main__":
    main()
