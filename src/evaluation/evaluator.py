"""Main evaluator for running models on benchmarks."""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .benchmarks import load_benchmark, BenchmarkSample
from .metrics import compute_accuracy, format_metrics


class ModelEvaluator:
    """Evaluator for running models on benchmarks."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        batch_size: int = 8,
        max_new_tokens: int = 512,
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to model checkpoint or HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or 'auto')
            batch_size: Batch size for inference
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_answer(self, question: str) -> str:
        """Generate an answer for a single question."""
        # Format prompt
        prompt = self._format_prompt(question)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        answer = generated_text[len(prompt):].strip()
        
        return answer
    
    def _format_prompt(self, question: str) -> str:
        """Format question as a prompt."""
        return f"""Solve the following problem step by step. Provide your final answer at the end.

Problem: {question}

Solution:"""
    
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        split: str = "test",
        output_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate model on a benchmark.
        
        Args:
            benchmark_name: Name of benchmark (gsm8k, math, mmlu)
            split: Dataset split to evaluate on
            output_dir: Directory to save results
            max_samples: Maximum number of samples to evaluate (for testing)
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating on {benchmark_name.upper()} ({split} split)")
        print(f"{'='*60}\n")
        
        # Load benchmark
        samples = load_benchmark(benchmark_name, split=split)
        
        if max_samples:
            samples = samples[:max_samples]
            print(f"Limiting to {max_samples} samples for testing")
        
        print(f"Loaded {len(samples)} samples")
        
        # Run evaluation
        predictions = []
        for sample in tqdm(samples, desc="Evaluating"):
            # Generate answer
            predicted_answer = self.generate_answer(sample.question)
            
            # Store prediction
            predictions.append({
                'id': sample.id,
                'question': sample.question,
                'target_answer': sample.answer,
                'predicted_answer': predicted_answer,
                'category': sample.category,
                'difficulty': sample.difficulty,
                'is_correct': None,  # Will be computed by metrics
            })
        
        # Compute metrics
        metrics = compute_accuracy(predictions)
        
        # Update is_correct in predictions
        from .metrics import is_correct
        for pred in predictions:
            pred['is_correct'] = is_correct(pred['predicted_answer'], pred['target_answer'])
        
        # Create results dictionary
        results = {
            'benchmark': benchmark_name,
            'split': split,
            'model': self.model_path,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(samples),
            'accuracy': metrics.accuracy,
            'num_correct': metrics.num_correct,
            'num_incorrect': metrics.num_incorrect,
            'per_category_accuracy': metrics.per_category_accuracy,
            'per_difficulty_accuracy': metrics.per_difficulty_accuracy,
            'predictions': predictions,
        }
        
        # Print metrics
        print(f"\n{format_metrics(metrics)}")
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            result_file = os.path.join(output_dir, f"{benchmark_name}_results.json")
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {result_file}")
        
        return results
    
    def evaluate_all(
        self,
        benchmarks: List[str],
        output_dir: str,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate model on multiple benchmarks.
        
        Args:
            benchmarks: List of benchmark names
            output_dir: Directory to save results
            max_samples: Maximum samples per benchmark (for testing)
        
        Returns:
            Dictionary mapping benchmark names to results
        """
        all_results = {}
        
        for benchmark in benchmarks:
            results = self.evaluate_benchmark(
                benchmark,
                output_dir=output_dir,
                max_samples=max_samples,
            )
            all_results[benchmark] = results
        
        # Create summary
        summary = {
            'model': self.model_path,
            'timestamp': datetime.now().isoformat(),
            'benchmarks': {
                name: {
                    'accuracy': results['accuracy'],
                    'num_samples': results['num_samples'],
                }
                for name, results in all_results.items()
            },
            'average_accuracy': sum(r['accuracy'] for r in all_results.values()) / len(all_results),
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_file}")
        
        return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a model on benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument("--benchmarks", type=str, default="gsm8k,math,mmlu", help="Comma-separated benchmark names")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per benchmark (for testing)")
    
    args = parser.parse_args()
    
    # Parse benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(',')]
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        batch_size=args.batch_size,
    )
    
    # Run evaluation
    evaluator.evaluate_all(
        benchmarks=benchmarks,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )
