"""Model evaluator for benchmarks."""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from .benchmarks import load_benchmark, BenchmarkSample
from .metrics import compute_accuracy, EvaluationMetrics, format_metrics

class ModelEvaluator:
    """Evaluates language models on reasoning benchmarks."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
        batch_size: int = 1,
        use_mlflow: bool = True,
        experiment_name: str = "metacog-reasoning",
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path or name of model on HuggingFace
            device: Device to run on (cuda/cpu)
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for evaluation
            use_mlflow: Whether to use MLflow tracking
            experiment_name: MLflow experiment name
        """
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.use_mlflow = use_mlflow
        
        # Setup MLflow
        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
            mlflow.set_tracking_uri("file:./mlruns")
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()
        
    def generate_answer(self, question: str) -> str:
        """Generate answer for a question."""
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
        full_response = generated_text[len(prompt):].strip()
        
        # Extract final answer, passing question for MC value mapping
        answer = self._extract_final_answer(full_response, question)
        
        return answer
    
    def _extract_final_answer(self, text: str, question: str = None) -> str:
        """Extract the final answer from model's reasoning."""
        # Check if this is a multiple choice question
        is_mc = question and bool(re.search(r'\n[A-D]\.', question))
        
        # First, try to extract multiple choice answer (A, B, C, D)
        mc_patterns = [
            r'(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-D])\)?',
            r'\b([A-D])\)\s*(?:is|appears|seems)',
            r'(?:select|choose)\s*\(?([A-D])\)?',
            r'\(?([A-D])\)?\s*(?:is correct|is the answer)',
        ]
        
        for pattern in mc_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Look for standalone letter at end
        match = re.search(r'\b([A-D])\b\.?\s*$', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # For MC questions: try to map computed value to answer letter
        if is_mc:
            # Extract option values from question
            option_values = {}
            for letter in ['A', 'B', 'C', 'D']:
                match = re.search(rf'{letter}\.\s*(.+?)(?:\n|$)', question)
                if match:
                    option_values[letter] = match.group(1).strip()
            
            # Extract the computed value from model output
            numbers = re.findall(r'([\d,\.]+)', text)
            if numbers:
                computed = numbers[-1].replace(',', '')
                # Try to match computed value to an option
                for letter, value in option_values.items():
                    # Clean the option value
                    clean_value = re.sub(r'[^\d\.\-]', '', value)
                    if clean_value == computed:
                        return letter
            
            # Last resort for MC: extract any letter A-D
            letters = re.findall(r'\b([A-D])\b', text, re.IGNORECASE)
            if letters:
                return letters[-1].upper()
        
        # Try numerical answer patterns (for non-MC like GSM8k)
        num_patterns = [
            r'(?:final answer|answer|result)\s*(?:is|:)?\s*([\d,\.]+)',
            r'=\s*([\d,\.]+)\s*(?:clips|dollars|items|people|hours|minutes|cents|pounds|feet|inches|meters)?\s*\.?\s*$',
            r'(?:total|altogether)\s*(?:is|:)?\s*([\d,\.]+)',
            r'(?:she|he|they|it)\s+(?:sold|has|had|made|earned)\s*([\d,\.]+)',
        ]
        
        for pattern in num_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')
        
        # Fallback: extract last number in text
        numbers = re.findall(r'([\d,\.]+)', text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        # Last resort: extract any single letter A-D
        letters = re.findall(r'\b([A-D])\b', text, re.IGNORECASE)
        if letters:
            return letters[-1].upper()
        
        return text
    
    def _generate_code(self, sample) -> str:
        """Generate code for HumanEval problems."""
        # Use the original prompt from the sample
        prompt = sample.metadata.get('prompt', sample.question)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate with more tokens for code
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the function body (after the prompt)
        code = generated_text[len(prompt):].strip()
        
        # Stop at end of function (double newline or next def)
        if '\n\ndef ' in code:
            code = code.split('\n\ndef ')[0]
        if '\n\nclass ' in code:
            code = code.split('\n\nclass ')[0]
        
        return code
    
    def _execute_code_test(self, sample, generated_code: str) -> bool:
        """Execute HumanEval test cases to check correctness."""
        try:
            # Combine prompt + generated code + test
            full_code = sample.metadata['prompt'] + generated_code + "\n" + sample.metadata['test']
            
            # Execute in isolated namespace
            exec_globals = {}
            exec(full_code, exec_globals)
            
            # If we get here without exception, tests passed
            return True
        except Exception as e:
            return False
    
    def _format_prompt(self, question: str) -> str:
        """Format question as a prompt."""
        # Detect if this is a multiple choice question
        is_multiple_choice = bool(re.search(r'\n[A-D]\.', question))
        
        if is_multiple_choice:
            return f"""Answer the following multiple choice question. Think step by step, then provide your final answer as a single letter (A, B, C, or D).

Question: {question}

Solution:"""
        else:
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
        # Start MLflow run
        if self.use_mlflow:
            run_name = f"{benchmark_name}_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name)
            
            # Log parameters
            mlflow.log_param("model", self.model_path)
            mlflow.log_param("benchmark", benchmark_name)
            mlflow.log_param("split", split)
            mlflow.log_param("max_new_tokens", self.max_new_tokens)
            mlflow.log_param("device", self.device)
            if max_samples:
                mlflow.log_param("max_samples", max_samples)
        
        try:
            # HellaSwag test split has no labels, use validation instead
            if benchmark_name.lower() == 'hellaswag' and split == 'test':
                split = 'validation'
            
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
        is_humaneval = benchmark_name.lower() == 'humaneval'
        
        for sample in tqdm(samples, desc="Evaluating"):
            # Generate answer
            if is_humaneval:
                # For HumanEval, use code generation prompt
                predicted_answer = self._generate_code(sample)
            else:
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
        for i, pred in enumerate(predictions):
            if is_humaneval:
                # For HumanEval, execute the code to check correctness
                pred['is_correct'] = self._execute_code_test(samples[i], pred['predicted_answer'])
            else:
                pred['is_correct'] = is_correct(pred['predicted_answer'], pred['target_answer'])
        
        # Recompute metrics after code execution for HumanEval
        if is_humaneval:
            num_correct = sum(1 for p in predictions if p['is_correct'])
            metrics = EvaluationMetrics(
                accuracy=num_correct / len(predictions) if predictions else 0,
                num_correct=num_correct,
                num_incorrect=len(predictions) - num_correct,
                per_category_accuracy={'code_generation': num_correct / len(predictions) if predictions else 0},
                per_difficulty_accuracy=None
            )
        
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
        
            # Print results
            print(f"\nAccuracy: {metrics.accuracy:.1%} ({metrics.num_correct}/{len(samples)})")
            print(f"Correct: {metrics.num_correct}")
            print(f"Incorrect: {metrics.num_incorrect}")
            
            # Log metrics to MLflow
            if self.use_mlflow:
                mlflow.log_metric("accuracy", metrics.accuracy)
                mlflow.log_metric("num_correct", metrics.num_correct)
                mlflow.log_metric("num_incorrect", metrics.num_incorrect)
                mlflow.log_metric("num_samples", len(samples))
            
            if metrics.per_category_accuracy:
                print("\nPer-Category Accuracy:")
                for category, acc in metrics.per_category_accuracy.items():
                    print(f"  {category}: {acc:.1%}")
                    if self.use_mlflow:
                        mlflow.log_metric(f"accuracy_{category}", acc)
            
            if metrics.per_difficulty_accuracy:
                print("\nPer-Difficulty Accuracy:")
                for difficulty, acc in sorted(metrics.per_difficulty_accuracy.items()):
                    print(f"  Level {difficulty}: {acc:.1%}")
                    if self.use_mlflow:
                        mlflow.log_metric(f"accuracy_level_{difficulty}", acc)
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save detailed results
            results_file = os.path.join(output_dir, f"{benchmark_name}_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {results_file}")
            
            # Save summary
            summary_file = os.path.join(output_dir, "summary.json")
            summary = {
                'model': self.model_path,
                'timestamp': results['timestamp'],
                'benchmarks': {
                    benchmark_name: {
                        'accuracy': metrics.accuracy,
                        'num_samples': len(samples),
                        'num_correct': metrics.num_correct,
                    }
                }
            }
            
            # Merge with existing summary if it exists
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    existing_summary = json.load(f)
                    existing_summary['benchmarks'].update(summary['benchmarks'])
                    summary = existing_summary
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to {summary_file}")
            
            # Log artifacts to MLflow
            if self.use_mlflow:
                mlflow.log_artifact(results_file)
                mlflow.log_artifact(summary_file)
        
            return results
        
        finally:
            # End MLflow run
            if self.use_mlflow:
                mlflow.end_run()
