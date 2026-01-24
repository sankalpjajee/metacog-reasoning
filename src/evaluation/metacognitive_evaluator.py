"""Metacognitive model evaluator for benchmarks.

This evaluator uses metacognitive prompting to test if explicit monitoring
and control improves model accuracy on reasoning tasks.
"""

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


class MetacognitiveEvaluator:
    """Evaluates language models with metacognitive prompting."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 2048,  # Longer for reasoning traces
        batch_size: int = 1,
        use_mlflow: bool = True,
        experiment_name: str = "metacog-reasoning-validation",
    ):
        """
        Initialize metacognitive evaluator.
        
        Args:
            model_path: Path or name of model on HuggingFace
            device: Device to run on (cuda/cpu)
            max_new_tokens: Maximum tokens to generate (longer for reasoning)
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
        
    def generate_answer(self, question: str, benchmark_name: str = None) -> tuple:
        """
        Generate answer for a question with metacognitive prompting.
        
        Returns:
            tuple: (full_response, extracted_answer)
        """
        # Format prompt with metacognitive instructions
        prompt = self._format_metacognitive_prompt(question, benchmark_name)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate with longer max_tokens for reasoning
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=False,  # Greedy for consistency
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode - keep special tokens to properly extract assistant response
        generated_text_with_tokens = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract answer (remove prompt)
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text_with_tokens:
            parts = generated_text_with_tokens.split("<|start_header_id|>assistant<|end_header_id|>")
            if len(parts) > 1:
                full_response = parts[-1].strip()
                # Remove trailing special tokens
                for token in ['<|eot_id|>', '<|end_of_text|>']:
                    full_response = full_response.replace(token, '')
                full_response = full_response.strip()
            else:
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                full_response = generated_text[len(prompt):].strip()
        else:
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            full_response = generated_text[len(prompt):].strip()
        
        # Extract final answer from reasoning trace
        answer = self._extract_final_answer(full_response, question, benchmark_name)
        
        return full_response, answer
    
    def _format_metacognitive_prompt(self, question: str, benchmark_name: str = None) -> str:
        """Format question with metacognitive prompting.
        
        Based on Wang & Zhao (NAACL 2024) "Metacognitive Prompting Improves Understanding in LLMs"
        Enhanced to align with Nelson & Narens (1990) framework with stronger control mechanisms.
        """
        
        # Simple system message
        system_message = """You are a helpful assistant that solves problems carefully and thoughtfully."""
        
        # Benchmark-specific user message formatting
        if benchmark_name and benchmark_name.lower() == 'gsm8k':
            user_message = f"""First, assess if this problem is SIMPLE or COMPLEX:
- SIMPLE: Basic arithmetic or 1-2 step calculation
- COMPLEX: Multi-step reasoning, word problems, or requires careful analysis

If SIMPLE: Solve directly and provide your final answer.

If COMPLEX: Follow these steps:

1. Clarify your understanding of what the problem is asking.

2. Make a preliminary solution to the problem.

3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.

4. Once you have a solution, decide whether you need additional verification or if you're confident enough to finalize.

5. Provide your final answer with a clear explanation of your reasoning.

6. Rate your overall confidence (0-100%) in this answer and explain why.

Problem: {question}

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [just the number, nothing else]

Example: If the answer is 25 dollars, write:
Final Answer: 25"""
        
        elif benchmark_name and benchmark_name.lower() in ['mmlu', 'hellaswag', 'mrben']:
            user_message = f"""First, assess if this question is SIMPLE or COMPLEX:
- SIMPLE: The answer is immediately clear from basic knowledge or intuition
- COMPLEX: Requires careful analysis, comparison of options, or multi-step reasoning

If SIMPLE: Choose the answer directly and provide your final answer.

If COMPLEX: Follow these steps:

1. Clarify your understanding of what the question is asking.

2. Make a preliminary analysis of each option.

3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.

4. Once you have selected an option, decide whether you need additional verification or if you're confident enough to finalize.

5. Provide your final answer with a clear explanation of your reasoning.

6. Rate your overall confidence (0-100%) in this answer and explain why.

{question}

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [just the letter A, B, C, or D]

Example: If you choose option B, write:
Final Answer: B"""
        
        else:
            # Generic format
            user_message = f"""First, assess if this problem is SIMPLE or COMPLEX:
- SIMPLE: Straightforward question with obvious answer
- COMPLEX: Requires careful reasoning or multi-step analysis

If SIMPLE: Solve directly and provide your final answer.

If COMPLEX: Follow these steps:

1. Clarify your understanding of what is being asked.

2. Make a preliminary solution to the problem.

3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.

4. Once you have a solution, decide whether you need additional verification or if you're confident enough to finalize.

5. Provide your final answer with a clear explanation of your reasoning.

6. Rate your overall confidence (0-100%) in this answer and explain why.

Problem: {question}

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [your answer]

Make sure to write "Final Answer:" followed by just your answer."""
        
        # Use Llama-3.1 Instruct chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def _extract_final_answer(self, text: str, question: str = None, benchmark_name: str = None) -> str:
        """Extract the final answer from metacognitive reasoning trace."""
        
        # PRIORITY 1: Look for "Final Answer: X" format (from our prompt)
        final_answer_match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if final_answer_match:
            answer = final_answer_match.group(1).strip()
            return self._normalize_answer(answer, benchmark_name)
        
        # PRIORITY 2: Look for other explicit answer markers
        answer_patterns = [
            r'(?:Therefore|Thus|So),?\s*(?:the answer is|answer:)\s*(.+?)(?:\n|$)',
            r'(?:The correct answer is|The answer is):\s*(.+?)(?:\n|$)',
            r'(?:Answer|ANSWER):\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                return self._normalize_answer(answer, benchmark_name)
        
        # PRIORITY 3: For multiple choice, look for letter patterns
        if benchmark_name and benchmark_name.lower() in ['mmlu', 'hellaswag', 'mrben']:
            mc_patterns = [
                r'(?:option|choice)\s*\(?([A-D])\)?',
                r'\b([A-D])\)\s*(?:is|appears)',
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
        
        # PRIORITY 4: For numerical answers (GSM8K)
        if benchmark_name and benchmark_name.lower() == 'gsm8k':
            num_patterns = [
                r'=\s*([\d,\.]+)\s*(?:\.|$)',
                r'(?:total|result)\s*(?:is|:)?\s*([\d,\.]+)',
            ]
            
            for pattern in num_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).replace(',', '')
            
            # Extract last number
            numbers = re.findall(r'([\d,\.]+)', text)
            if numbers:
                return numbers[-1].replace(',', '')
        
        # FALLBACK: Return last line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            last_line = lines[-1]
            # Remove common prefixes
            last_line = re.sub(r'^(?:Answer|Therefore|Thus|So)[:\s]+', '', last_line, flags=re.IGNORECASE)
            return self._normalize_answer(last_line, benchmark_name)
        
        return text[:100]  # Return first 100 chars as fallback
    
    def _normalize_answer(self, answer: str, benchmark_name: str = None) -> str:
        """Normalize answer for comparison."""
        answer = answer.strip().strip('"\'.,;')
        
        # For multiple choice, extract just the letter
        if benchmark_name and benchmark_name.lower() in ['mmlu', 'hellaswag', 'mrben']:
            match = re.match(r'^([A-D])', answer.upper())
            if match:
                return match.group(1)
            # Look for letter anywhere in answer
            match = re.search(r'\b([A-D])\b', answer.upper())
            if match:
                return match.group(1)
        
        # For numerical answers, remove commas
        if benchmark_name and benchmark_name.lower() == 'gsm8k':
            answer = answer.replace(',', '')
            # Extract just the number
            match = re.search(r'([\d\.]+)', answer)
            if match:
                return match.group(1)
        
        return answer
    
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        split: str = "test",
        output_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        save_traces: bool = True,
    ) -> Dict:
        """
        Evaluate model on a benchmark with metacognitive prompting.
        
        Args:
            benchmark_name: Name of benchmark (gsm8k, mmlu, hellaswag, mrben)
            split: Dataset split to evaluate on
            output_dir: Directory to save results
            max_samples: Maximum number of samples to evaluate
            save_traces: Whether to save full reasoning traces
        
        Returns:
            Dictionary with evaluation results
        """
        # Start MLflow run
        if self.use_mlflow:
            run_name = f"metacog_{benchmark_name}_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name)
            
            # Log parameters
            mlflow.log_param("model", self.model_path)
            mlflow.log_param("benchmark", benchmark_name)
            mlflow.log_param("split", split)
            mlflow.log_param("max_new_tokens", self.max_new_tokens)
            mlflow.log_param("evaluation_type", "metacognitive")
            if max_samples:
                mlflow.log_param("max_samples", max_samples)
        
        try:
            # HellaSwag test split has no labels, use validation instead
            if benchmark_name.lower() == 'hellaswag' and split == 'test':
                split = 'validation'
            
            print(f"\n{'='*60}")
            print(f"METACOGNITIVE Evaluation on {benchmark_name.upper()} ({split})")
            print(f"{'='*60}\n")
            
            # Load benchmark
            samples = load_benchmark(benchmark_name, split=split)
            
            if max_samples:
                samples = samples[:max_samples]
            
            print(f"Loaded {len(samples)} samples\n")
            
            # Evaluate
            results = []
            correct = 0
            total = 0
            
            for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {benchmark_name}")):
                try:
                    # Generate answer with metacognitive prompting
                    full_response, predicted_answer = self.generate_answer(
                        sample.question,
                        benchmark_name
                    )
                    
                    # Check correctness
                    is_correct = self._check_answer(
                        predicted_answer,
                        sample.answer,
                        benchmark_name
                    )
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    # Store result
                    result = {
                        'sample_id': i,
                        'question': sample.question,
                        'ground_truth': sample.answer,
                        'predicted': predicted_answer,
                        'correct': is_correct,
                    }
                    
                    if save_traces:
                        result['full_response'] = full_response
                    
                    results.append(result)
                    
                    # Print progress every 100 samples
                    if (i + 1) % 100 == 0:
                        current_acc = correct / total * 100
                        print(f"\nProgress: {i+1}/{len(samples)} - Current accuracy: {current_acc:.2f}%")
                
                except Exception as e:
                    print(f"\nError on sample {i}: {e}")
                    results.append({
                        'sample_id': i,
                        'question': sample.question,
                        'error': str(e),
                        'correct': False
                    })
                    total += 1
            
            # Calculate final metrics
            accuracy = correct / total * 100 if total > 0 else 0.0
            
            print(f"\n{'='*60}")
            print(f"RESULTS: {benchmark_name.upper()}")
            print(f"{'='*60}")
            print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
            print(f"{'='*60}\n")
            
            # Save results
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save detailed results
                results_file = os.path.join(
                    output_dir,
                    f"metacog_{benchmark_name}_{split}_results.json"
                )
                with open(results_file, 'w') as f:
                    json.dump({
                        'benchmark': benchmark_name,
                        'split': split,
                        'model': self.model_path,
                        'evaluation_type': 'metacognitive',
                        'accuracy': accuracy,
                        'correct': correct,
                        'total': total,
                        'results': results
                    }, f, indent=2)
                
                print(f"Results saved to: {results_file}\n")
            
            # Log to MLflow
            if self.use_mlflow:
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("correct", correct)
                mlflow.log_metric("total", total)
            
            return {
                'benchmark': benchmark_name,
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'results': results
            }
        
        finally:
            if self.use_mlflow:
                mlflow.end_run()
    
    def _check_answer(self, predicted: str, ground_truth: str, benchmark_name: str = None) -> bool:
        """Check if predicted answer matches ground truth."""
        # Normalize both
        pred_norm = predicted.strip().lower()
        gt_norm = ground_truth.strip().lower()
        
        # Exact match
        if pred_norm == gt_norm:
            return True
        
        # For numerical answers, check numerical equivalence
        if benchmark_name and benchmark_name.lower() == 'gsm8k':
            try:
                pred_num = float(re.sub(r'[^\d\.\-]', '', predicted))
                gt_num = float(re.sub(r'[^\d\.\-]', '', ground_truth))
                return abs(pred_num - gt_num) < 0.01
            except (ValueError, AttributeError):
                pass
        
        # For multiple choice, just compare letters
        if benchmark_name and benchmark_name.lower() in ['mmlu', 'hellaswag', 'mrben']:
            return pred_norm.upper() == gt_norm.upper()
        
        return False


def main():
    """Run metacognitive evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model with metacognitive prompting")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--benchmarks", nargs="+", required=True, help="Benchmarks to evaluate")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples per benchmark")
    parser.add_argument("--output_dir", type=str, default="results/metacognitive", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow tracking")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MetacognitiveEvaluator(
        model_path=args.model_name,
        device=args.device,
        use_mlflow=not args.no_mlflow,
    )
    
    # Run evaluation on each benchmark
    all_results = {}
    for benchmark in args.benchmarks:
        result = evaluator.evaluate_benchmark(
            benchmark_name=benchmark,
            split="test",
            output_dir=args.output_dir,
            max_samples=args.num_samples,
        )
        all_results[benchmark] = result['accuracy']
    
    # Print summary
    print(f"\n{'='*60}")
    print("METACOGNITIVE EVALUATION SUMMARY")
    print(f"{'='*60}")
    for benchmark, accuracy in all_results.items():
        print(f"{benchmark:15s}: {accuracy:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
