"""
Self-Consistency Based Selective Metacognition Evaluator V2

FIXED VERSION: Uses proper chat template and answer extraction matching the working evaluators.

This evaluator implements selective metacognition using self-consistency as an uncertainty signal.
It generates multiple baseline answers and checks agreement. If agreement is high, it uses the
majority answer. If agreement is low, it applies full metacognitive reasoning.

Based on: Wang et al. (2022) "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
"""

import argparse
import json
import os
import re
import torch
from collections import Counter
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.evaluation.metacognitive_evaluator import MetacognitiveEvaluator
from src.evaluation.benchmarks import load_benchmark


class SelfConsistencyEvaluatorV2(MetacognitiveEvaluator):
    """Evaluator that uses self-consistency to decide when to apply metacognition. V2 with fixes."""
    
    def __init__(self, model_name: str, n_samples: int = 3, agreement_threshold: float = 0.67, 
                 temperature: float = 0.7):
        """
        Initialize the self-consistency evaluator.
        
        Args:
            model_name: Name of the model to evaluate
            n_samples: Number of samples to generate for self-consistency check
            agreement_threshold: Threshold for agreement (0-1). If agreement >= threshold, use baseline.
            temperature: Temperature for sampling diverse answers
        """
        super().__init__(model_name, device="cuda", max_new_tokens=1024)
        self.n_samples = n_samples
        self.agreement_threshold = agreement_threshold
        self.temperature = temperature
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _format_baseline_prompt(self, question: str, benchmark_name: str) -> str:
        """Format baseline prompt using proper chat template - SAME as working evaluators."""
        
        system_message = "You are a helpful assistant that solves problems carefully."
        
        if benchmark_name == "gsm8k":
            user_message = f"""Solve this math problem step by step.

Problem: {question}

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [just the number, nothing else]

Example: If the answer is 25 dollars, write:
Final Answer: 25"""
        
        elif benchmark_name == "mmlu":
            user_message = f"""Answer this multiple choice question.

{question}

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [just the letter A, B, C, or D]

Example: If you choose option B, write:
Final Answer: B"""
        
        elif benchmark_name == "hellaswag":
            user_message = f"""Complete this scenario by selecting the most plausible continuation.

{question}

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [just the number 0, 1, 2, or 3]

Example: If you choose option 1, write:
Final Answer: 1"""
        
        elif benchmark_name == "mrben":
            user_message = f"""Answer this multi-step reasoning question.

{question}

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [your answer]"""
        
        else:
            user_message = f"""Answer this question.

{question}

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [your answer]"""
        
        # Use Llama-3.1 Instruct chat template - SAME as working evaluators
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def _format_metacognitive_prompt(self, question: str, benchmark_name: str) -> str:
        """Format metacognitive prompt - SAME as working evaluators."""
        
        system_message = "You are a helpful assistant that solves problems carefully and thoughtfully."
        
        if benchmark_name == "gsm8k":
            user_message = f"""Solve this math problem using careful metacognitive reasoning.

Problem: {question}

Follow these steps:
1. Clarify your understanding of what the problem is asking.
2. Make a preliminary solution to the problem.
3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
4. Once you have a solution, decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation of your reasoning.
6. Rate your overall confidence (0-100%) in this answer and explain why.

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [just the number, nothing else]

Example: If the answer is 25 dollars, write:
Final Answer: 25"""
        
        elif benchmark_name == "mmlu":
            user_message = f"""Answer this multiple choice question using careful metacognitive reasoning.

{question}

Follow these steps:
1. Clarify your understanding of what the question is asking.
2. Make a preliminary analysis of each option.
3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
4. Once you have selected an option, decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation of your reasoning.
6. Rate your overall confidence (0-100%) in this answer and explain why.

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [just the letter A, B, C, or D]

Example: If you choose option B, write:
Final Answer: B"""
        
        elif benchmark_name == "hellaswag":
            user_message = f"""Select the most plausible continuation using careful metacognitive reasoning.

{question}

Follow these steps:
1. Clarify your understanding of the scenario.
2. Make a preliminary analysis of each option.
3. Monitor your confidence in the selection. If you think it's wrong or has potential errors, pause and verify your work.
4. Once you have selected an option, decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation of your reasoning.
6. Rate your overall confidence (0-100%) in this answer and explain why.

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [just the number 0, 1, 2, or 3]

Example: If you choose option 1, write:
Final Answer: 1"""
        
        elif benchmark_name == "mrben":
            user_message = f"""Answer this multi-step reasoning question using careful metacognitive reasoning.

{question}

Follow these steps:
1. Clarify your understanding of what the question is asking.
2. Make a preliminary solution to the problem.
3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
4. Once you have a solution, decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation of your reasoning.
6. Rate your overall confidence (0-100%) in this answer and explain why.

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [your answer]"""
        
        else:
            user_message = f"""Answer this question using careful metacognitive reasoning.

{question}

Follow these steps:
1. Clarify your understanding of what the question is asking.
2. Make a preliminary analysis.
3. Monitor your confidence. If you think it's wrong or has potential errors, pause and verify your work.
4. Decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation.
6. Rate your overall confidence (0-100%) and explain why.

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [your answer]"""
        
        # Use Llama-3.1 Instruct chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def _extract_response(self, generated_text: str) -> str:
        """Extract response from generated text - SAME as working evaluators."""
        
        # Try to extract assistant response from chat template
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            parts = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")
            if len(parts) > 1:
                response = parts[-1].strip()
                # Remove trailing special tokens
                for token in ['<|eot_id|>', '<|end_of_text|>']:
                    response = response.replace(token, '')
                return response.strip()
        
        return generated_text
        
    def generate_multiple_answers(self, question: str, benchmark_name: str) -> List[str]:
        """
        Generate multiple answers using baseline prompt with sampling.
        
        Args:
            question: The question to answer
            benchmark_name: Name of the benchmark (for prompt formatting)
            
        Returns:
            List of generated answers
        """
        answers = []
        
        # Use baseline prompt with proper chat template
        prompt = self._format_baseline_prompt(question, benchmark_name)
        
        for _ in range(self.n_samples):
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode with special tokens to properly extract
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract response using same method as working evaluators
            response = self._extract_response(generated_text)
            
            # Extract answer using parent class method
            answer = self._extract_final_answer(response, question, benchmark_name)
            answers.append(answer)
            
        return answers
    
    def compute_agreement(self, answers: List[str]) -> Tuple[float, str]:
        """
        Compute agreement rate and majority answer.
        Filter out empty/garbage answers before computing.
        
        Args:
            answers: List of answers
            
        Returns:
            Tuple of (agreement_rate, majority_answer)
        """
        if not answers:
            return 0.0, ""
        
        # Filter out empty and garbage answers
        valid_answers = [a for a in answers if a and a.strip() and a not in ['.', ',', '-', '']]
        
        if not valid_answers:
            return 0.0, answers[0] if answers else ""
        
        # Count occurrences
        counter = Counter(valid_answers)
        majority_answer, majority_count = counter.most_common(1)[0]
        
        # Compute agreement rate based on valid answers
        agreement_rate = majority_count / len(valid_answers)
        
        return agreement_rate, majority_answer
    
    def evaluate_single(self, sample, benchmark_name: str) -> Dict:
        """
        Evaluate a single sample using self-consistency based selective metacognition.
        
        Args:
            sample: The sample to evaluate (BenchmarkSample or dict)
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary with evaluation results
        """
        # Handle both BenchmarkSample objects and dictionaries
        if hasattr(sample, 'question'):
            question = sample.question
            ground_truth = sample.answer
        else:
            question = sample.get("question", sample.get("input", ""))
            ground_truth = sample.get("answer", sample.get("target", ""))
        
        # Step 1: Generate multiple baseline answers
        baseline_answers = self.generate_multiple_answers(question, benchmark_name)
        
        # Step 2: Compute agreement (with filtering)
        agreement_rate, majority_answer = self.compute_agreement(baseline_answers)
        
        # Step 3: Decide whether to apply metacognition
        if agreement_rate >= self.agreement_threshold:
            # High agreement → use majority answer (baseline)
            final_answer = majority_answer
            method_used = "baseline"
            full_response = f"Baseline answers: {baseline_answers}\nMajority answer: {majority_answer}"
        else:
            # Low agreement → apply metacognition
            metacog_prompt = self._format_metacognitive_prompt(question, benchmark_name)
            
            inputs = self.tokenizer(metacog_prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode with special tokens to properly extract
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            metacog_response = self._extract_response(generated_text)
            
            # Extract answer using parent class method
            final_answer = self._extract_final_answer(metacog_response, question, benchmark_name)
            method_used = "metacognition"
            full_response = f"Baseline answers: {baseline_answers}\nAgreement: {agreement_rate:.2f}\nMetacognitive response: {metacog_response}"
        
        # Check correctness using parent class method
        is_correct = self._check_answer(final_answer, ground_truth, benchmark_name)
        
        return {
            "question": question,
            "ground_truth": ground_truth,
            "baseline_answers": baseline_answers,
            "agreement_rate": agreement_rate,
            "majority_answer": majority_answer,
            "method_used": method_used,
            "final_answer": final_answer,
            "correct": is_correct,
            "full_response": full_response
        }
    
    def evaluate_benchmark(self, benchmark_name: str, num_samples: int = None) -> Dict:
        """
        Evaluate a benchmark using self-consistency based selective metacognition.
        
        Args:
            benchmark_name: Name of the benchmark to evaluate
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"SELF-CONSISTENCY V2 EVALUATION: {benchmark_name.upper()}")
        print(f"{'='*60}")
        print(f"N samples: {self.n_samples}")
        print(f"Agreement threshold: {self.agreement_threshold}")
        print(f"Temperature: {self.temperature}")
        
        # Load benchmark (load all, then slice if needed)
        # HellaSwag test split has no labels, use validation instead
        split = "validation" if benchmark_name.lower() == "hellaswag" else "test"
        samples = load_benchmark(benchmark_name, split=split)
        
        if num_samples:
            samples = samples[:num_samples]
            print(f"Limiting to {num_samples} samples")
        
        print(f"Loaded {len(samples)} samples")
        
        results = []
        correct = 0
        baseline_count = 0
        metacog_count = 0
        baseline_correct = 0
        metacog_correct = 0
        total_agreement = 0
        
        for sample in tqdm(samples, desc=f"Evaluating {benchmark_name}"):
            result = self.evaluate_single(sample, benchmark_name)
            results.append(result)
            
            if result["correct"]:
                correct += 1
            
            if result["method_used"] == "baseline":
                baseline_count += 1
                if result["correct"]:
                    baseline_correct += 1
            else:
                metacog_count += 1
                if result["correct"]:
                    metacog_correct += 1
            
            total_agreement += result["agreement_rate"]
        
        # Compute statistics
        total = len(results)
        accuracy = correct / total * 100 if total > 0 else 0
        avg_agreement = total_agreement / total if total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {benchmark_name.upper()}")
        print(f"{'='*60}")
        print(f"Overall accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"\nMethod distribution:")
        print(f"  Baseline: {baseline_count} ({baseline_count/total*100:.1f}%)")
        print(f"  Metacognition: {metacog_count} ({metacog_count/total*100:.1f}%)")
        print(f"\nAccuracy by method:")
        if baseline_count > 0:
            print(f"  Baseline: {baseline_correct/baseline_count*100:.2f}% ({baseline_correct}/{baseline_count})")
        if metacog_count > 0:
            print(f"  Metacognition: {metacog_correct/metacog_count*100:.2f}% ({metacog_correct}/{metacog_count})")
        print(f"\nAverage agreement rate: {avg_agreement:.2f}")
        print(f"{'='*60}")
        
        return {
            "benchmark": benchmark_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "baseline_count": baseline_count,
            "metacog_count": metacog_count,
            "baseline_correct": baseline_correct,
            "metacog_correct": metacog_correct,
            "avg_agreement": avg_agreement,
            "n_samples": self.n_samples,
            "agreement_threshold": self.agreement_threshold,
            "temperature": self.temperature,
            "results": results
        }


def main():
    parser = argparse.ArgumentParser(description="Self-Consistency Selective Metacognition Evaluator V2")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--benchmarks", type=str, default="gsm8k",
                        help="Comma-separated list of benchmarks to evaluate")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of samples for self-consistency check")
    parser.add_argument("--agreement_threshold", type=float, default=0.67,
                        help="Agreement threshold for using baseline")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--output_dir", type=str, default="results/selfconsistency_v2",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    print(f"Loading model: {args.model_name}")
    evaluator = SelfConsistencyEvaluatorV2(
        args.model_name,
        n_samples=args.n_samples,
        agreement_threshold=args.agreement_threshold,
        temperature=args.temperature
    )
    
    # Evaluate each benchmark
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    
    for benchmark in benchmarks:
        result = evaluator.evaluate_benchmark(benchmark, args.num_samples)
        
        # Save results
        output_file = os.path.join(args.output_dir, f"selfconsistency_v2_{benchmark}_results.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
