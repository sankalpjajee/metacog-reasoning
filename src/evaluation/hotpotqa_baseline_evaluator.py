"""Baseline evaluator for HotPotQA benchmark."""
import json
import os
from typing import Dict, List
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hotpotqa_loader import load_hotpotqa


class HotPotQABaselineEvaluator:
    """Simple baseline evaluator for HotPotQA."""
    
    def __init__(self,
                 model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the evaluator.
        
        Args:
            model_path: Path to the model
            device: Device to run on
        """
        self.device = device
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def evaluate(self,
                 num_samples: int = 100,
                 split: str = "validation",
                 output_dir: str = "results/hotpotqa_baseline") -> Dict:
        """Evaluate on HotPotQA.
        
        Args:
            num_samples: Number of samples to evaluate
            split: Dataset split
            output_dir: Output directory
        
        Returns:
            Dictionary with results
        """
        # Load dataset
        print(f"\nLoading HotPotQA {split} split...")
        samples = load_hotpotqa(split=split, setting="distractor", max_samples=num_samples)
        
        print(f"Evaluating {len(samples)} samples...")
        
        results = []
        correct = 0
        
        for sample in tqdm(samples, desc="Evaluating"):
            # Generate answer
            predicted_answer = self.answer_question(sample.question)
            
            # Check correctness
            is_correct = self.check_answer(predicted_answer, sample.answer)
            
            if is_correct:
                correct += 1
            
            results.append({
                'id': sample.id,
                'question': sample.question,
                'ground_truth': sample.answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'type': sample.metadata.get('type', 'unknown'),
                'level': sample.metadata.get('level', 'unknown')
            })
        
        # Calculate metrics
        accuracy = correct / len(samples) if samples else 0
        
        # Breakdown by type
        bridge_correct = sum(1 for r in results if r['type'] == 'bridge' and r['is_correct'])
        bridge_total = sum(1 for r in results if r['type'] == 'bridge')
        comparison_correct = sum(1 for r in results if r['type'] == 'comparison' and r['is_correct'])
        comparison_total = sum(1 for r in results if r['type'] == 'comparison')
        
        summary = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(samples),
            'bridge_accuracy': bridge_correct / bridge_total if bridge_total > 0 else 0,
            'bridge_total': bridge_total,
            'comparison_accuracy': comparison_correct / comparison_total if comparison_total > 0 else 0,
            'comparison_total': comparison_total,
        }
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'baseline_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(os.path.join(output_dir, 'baseline_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        self.print_summary(summary)
        
        return summary
    
    def answer_question(self, question: str) -> str:
        """Answer a question using the model.
        
        Args:
            question: The question
        
        Returns:
            The answer
        """
        # Simple baseline prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions concisely and accurately."},
            {"role": "user", "content": f"Question: {question}\n\nProvide a brief, direct answer:"}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer (after the prompt)
        answer = full_response[len(prompt):].strip()
        
        # Take first line as answer
        answer = answer.split('\n')[0].strip()
        
        return answer
    
    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
        
        Returns:
            True if correct
        """
        # Normalize
        pred_norm = predicted.lower().strip()
        gt_norm = ground_truth.lower().strip()
        
        # Exact match
        if pred_norm == gt_norm:
            return True
        
        # Containment (either direction)
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return True
        
        # Token overlap (for multi-word answers)
        pred_tokens = set(pred_norm.split())
        gt_tokens = set(gt_norm.split())
        
        if len(gt_tokens) > 0:
            overlap = len(pred_tokens & gt_tokens) / len(gt_tokens)
            if overlap >= 0.5:  # At least 50% token overlap
                return True
        
        return False
    
    def print_summary(self, summary: Dict):
        """Print evaluation summary.
        
        Args:
            summary: Summary dictionary
        """
        print("\n" + "="*60)
        print("HOTPOTQA BASELINE EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {summary['accuracy']*100:.2f}% ({summary['correct']}/{summary['total']})")
        print()
        print(f"Bridge Questions: {summary['bridge_accuracy']*100:.2f}% ({summary['bridge_total']} samples)")
        print(f"Comparison Questions: {summary['comparison_accuracy']*100:.2f}% ({summary['comparison_total']} samples)")
        print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline evaluation on HotPotQA")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Path to the model")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split (train or validation)")
    parser.add_argument("--output_dir", type=str, default="results/hotpotqa_baseline",
                        help="Output directory")
    
    args = parser.parse_args()
    
    evaluator = HotPotQABaselineEvaluator(model_path=args.model_path)
    evaluator.evaluate(
        num_samples=args.num_samples,
        split=args.split,
        output_dir=args.output_dir
    )
