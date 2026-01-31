"""
Enhanced Error Prediction Data Generation (V2)

Generates training data for error prediction probe with:
- Multi-layer hidden states (layers 8, 16, 24, 32)
- Baseline confidence features (entropy, logits, probabilities)
- Self-consistency agreement rate (3 samples)
- Ground truth labels (correct/wrong)

Target probe accuracy: 80%+
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.benchmarks import load_benchmark


class EnhancedDataGenerator:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading model: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            output_hidden_states=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        
        # Layers to extract hidden states from
        self.layers = [8, 16, 24, 32]
        
    def extract_multi_layer_hidden_states(self, question: str) -> torch.Tensor:
        """Extract hidden states from multiple layers."""
        # Tokenize
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract hidden states from specified layers
        hidden_states = []
        for layer_idx in self.layers:
            # Get last token's hidden state from this layer
            layer_hidden = outputs.hidden_states[layer_idx][0, -1, :]  # [hidden_dim]
            hidden_states.append(layer_hidden)
        
        # Concatenate all layers
        combined_hidden = torch.cat(hidden_states, dim=0)  # [num_layers * hidden_dim]
        return combined_hidden.cpu()
    
    def generate_baseline_with_confidence(self, question: str, temperature: float = 0.7) -> Tuple[str, Dict]:
        """Generate baseline answer and extract confidence features."""
        # Format prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions concisely and accurately."},
            {"role": "user", "content": f"{question}\n\nProvide your answer:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with logits
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Extract answer
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Extract confidence features from logits
        # Average over all generated tokens
        all_logits = torch.stack(outputs.scores, dim=0)  # [seq_len, batch, vocab_size]
        all_probs = torch.softmax(all_logits, dim=-1)
        
        # Confidence features
        max_probs = all_probs.max(dim=-1).values  # [seq_len, batch]
        top2_probs = torch.topk(all_probs, k=2, dim=-1).values  # [seq_len, batch, 2]
        
        confidence_features = {
            'mean_max_prob': max_probs.mean().item(),
            'min_max_prob': max_probs.min().item(),
            'mean_entropy': -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1).mean().item(),
            'mean_top1_top2_gap': (top2_probs[:, :, 0] - top2_probs[:, :, 1]).mean().item(),
        }
        
        return answer, confidence_features
    
    def compute_self_consistency(self, question: str, n_samples: int = 3) -> Tuple[float, List[str]]:
        """Generate multiple samples and compute agreement rate."""
        answers = []
        for _ in range(n_samples):
            answer, _ = self.generate_baseline_with_confidence(question, temperature=0.7)
            answers.append(self._extract_final_answer(answer))
        
        # Compute agreement rate
        if not answers or all(a == "" for a in answers):
            return 0.0, answers
        
        # Count most common answer
        from collections import Counter
        answer_counts = Counter(answers)
        most_common_count = answer_counts.most_common(1)[0][1]
        agreement_rate = most_common_count / len(answers)
        
        return agreement_rate, answers
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract final answer from generated text."""
        text = text.strip()
        
        # Look for "Final Answer:" pattern
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].strip()
            # Remove any trailing explanation
            answer = answer.split("\n")[0].strip()
            return answer
        
        # Look for "Answer:" pattern
        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip()
            answer = answer.split("\n")[0].strip()
            return answer
        
        # Take first line as answer
        return text.split("\n")[0].strip()
    
    def _check_answer(self, predicted: str, ground_truth: str, benchmark_name: str) -> bool:
        """Check if predicted answer matches ground truth."""
        pred_norm = predicted.strip().lower()
        gt_norm = ground_truth.strip().lower()
        
        # Multiple choice (MMLU, HellaSwag)
        if benchmark_name.lower() in ['mmlu', 'hellaswag']:
            return pred_norm.upper() == gt_norm.upper()
        
        # Numeric (GSM8K)
        # Try exact match first
        if pred_norm == gt_norm:
            return True
        
        # Try numeric comparison
        try:
            pred_num = float(pred_norm.replace(',', ''))
            gt_num = float(gt_norm.replace(',', ''))
            return abs(pred_num - gt_num) < 1e-6
        except:
            pass
        
        # Containment
        return pred_norm in gt_norm or gt_norm in pred_norm
    
    def generate_training_data(
        self,
        benchmark_name: str,
        num_samples: int,
        output_dir: str,
    ):
        """Generate training data for a benchmark."""
        print(f"\n{'='*60}")
        print(f"Generating training data for {benchmark_name.upper()}")
        print(f"{'='*60}")
        
        # Load benchmark
        print(f"Loading {benchmark_name}...")
        if benchmark_name.lower() == 'gsm8k':
            samples = load_benchmark(benchmark_name, split="train")
        elif benchmark_name.lower() == 'mmlu':
            samples = load_benchmark(benchmark_name, split="dev")  # MMLU uses dev for training
        elif benchmark_name.lower() == 'hellaswag':
            samples = load_benchmark(benchmark_name, split="train")
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        # Stratified sampling for MMLU
        if benchmark_name.lower() == 'mmlu':
            samples = self._stratified_sample_mmlu(samples, num_samples)
        else:
            # Random sample
            if len(samples) > num_samples:
                samples = random.sample(samples, num_samples)
        
        print(f"Processing {len(samples)} samples...")
        
        # Generate data
        training_data = []
        for sample in tqdm(samples, desc=f"Processing {benchmark_name}"):
            try:
                # Extract question and ground truth
                question = sample['question']
                ground_truth = sample['answer']
                
                # 1. Extract multi-layer hidden states
                hidden_states = self.extract_multi_layer_hidden_states(question)
                
                # 2. Generate baseline answer with confidence
                baseline_answer, confidence_features = self.generate_baseline_with_confidence(question)
                baseline_answer_extracted = self._extract_final_answer(baseline_answer)
                
                # 3. Compute self-consistency
                agreement_rate, consistency_answers = self.compute_self_consistency(question, n_samples=3)
                
                # 4. Check correctness
                is_correct = self._check_answer(baseline_answer_extracted, ground_truth, benchmark_name)
                
                # Store data
                training_data.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'baseline_answer': baseline_answer_extracted,
                    'is_correct': is_correct,
                    'hidden_states': hidden_states,
                    'confidence_features': confidence_features,
                    'agreement_rate': agreement_rate,
                    'consistency_answers': consistency_answers,
                    'benchmark': benchmark_name,
                })
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        # Save data
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tensors
        tensor_data = {
            'hidden_states': torch.stack([d['hidden_states'] for d in training_data]),
            'labels': torch.tensor([0 if d['is_correct'] else 1 for d in training_data], dtype=torch.float32),
            'confidence_features': torch.tensor([
                [d['confidence_features']['mean_max_prob'],
                 d['confidence_features']['min_max_prob'],
                 d['confidence_features']['mean_entropy'],
                 d['confidence_features']['mean_top1_top2_gap']]
                for d in training_data
            ], dtype=torch.float32),
            'agreement_rates': torch.tensor([d['agreement_rate'] for d in training_data], dtype=torch.float32),
        }
        
        tensor_path = os.path.join(output_dir, f"{benchmark_name}_tensors.pt")
        torch.save(tensor_data, tensor_path)
        print(f"Saved tensors to: {tensor_path}")
        
        # Save metadata
        metadata = [{
            'question': d['question'],
            'ground_truth': d['ground_truth'],
            'baseline_answer': d['baseline_answer'],
            'is_correct': d['is_correct'],
            'agreement_rate': d['agreement_rate'],
            'consistency_answers': d['consistency_answers'],
            'benchmark': d['benchmark'],
        } for d in training_data]
        
        metadata_path = os.path.join(output_dir, f"{benchmark_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_path}")
        
        # Print statistics
        num_correct = sum(1 for d in training_data if d['is_correct'])
        num_wrong = len(training_data) - num_correct
        print(f"\nStatistics:")
        print(f"  Total samples: {len(training_data)}")
        print(f"  Correct: {num_correct} ({num_correct/len(training_data)*100:.1f}%)")
        print(f"  Wrong: {num_wrong} ({num_wrong/len(training_data)*100:.1f}%)")
        print(f"  Mean agreement rate: {sum(d['agreement_rate'] for d in training_data) / len(training_data):.3f}")
        
        return training_data
    
    def _stratified_sample_mmlu(self, samples: List[Dict], num_samples: int) -> List[Dict]:
        """Stratified sampling for MMLU across subjects."""
        # Group by subject
        from collections import defaultdict
        by_subject = defaultdict(list)
        for sample in samples:
            subject = sample.get('subject', 'unknown')
            by_subject[subject].append(sample)
        
        # Sample proportionally from each subject
        samples_per_subject = num_samples // len(by_subject)
        stratified_samples = []
        
        for subject, subject_samples in by_subject.items():
            if len(subject_samples) <= samples_per_subject:
                stratified_samples.extend(subject_samples)
            else:
                stratified_samples.extend(random.sample(subject_samples, samples_per_subject))
        
        # If we didn't get enough, sample more randomly
        if len(stratified_samples) < num_samples:
            remaining = num_samples - len(stratified_samples)
            all_samples = [s for subj_samples in by_subject.values() for s in subj_samples]
            available = [s for s in all_samples if s not in stratified_samples]
            stratified_samples.extend(random.sample(available, min(remaining, len(available))))
        
        return stratified_samples[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced error prediction training data")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "hellaswag"])
    parser.add_argument("--samples_per_benchmark", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="data/training/error_prediction_v2")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create generator
    generator = EnhancedDataGenerator(args.model_path)
    
    # Generate data for each benchmark
    for benchmark in args.benchmarks:
        generator.generate_training_data(
            benchmark_name=benchmark,
            num_samples=args.samples_per_benchmark,
            output_dir=args.output_dir,
        )
    
    print(f"\n{'='*60}")
    print("Data generation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
