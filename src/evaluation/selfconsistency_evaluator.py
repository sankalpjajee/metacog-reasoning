"""
Self-Consistency Based Selective Metacognition Evaluator

This evaluator implements selective metacognition using self-consistency as an uncertainty signal.
It generates multiple baseline answers and checks agreement. If agreement is high, it uses the
majority answer. If agreement is low, it applies full metacognitive reasoning.

Based on: Wang et al. (2022) "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.evaluation.base_evaluator import BaseEvaluator


class SelfConsistencyEvaluator(BaseEvaluator):
    """Evaluator that uses self-consistency to decide when to apply metacognition."""
    
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
        super().__init__(model_name)
        self.n_samples = n_samples
        self.agreement_threshold = agreement_threshold
        self.temperature = temperature
        
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
        
        # Use baseline prompt (no metacognition)
        prompt = self._format_baseline_prompt(question, benchmark_name)
        
        for _ in range(self.n_samples):
            response = self.model.generate(
                prompt,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95
            )
            
            # Extract answer from response
            answer = self.extract_answer(response, benchmark_name)
            answers.append(answer)
            
        return answers
    
    def compute_agreement(self, answers: List[str]) -> Tuple[float, str]:
        """
        Compute agreement rate and majority answer.
        
        Args:
            answers: List of answers
            
        Returns:
            Tuple of (agreement_rate, majority_answer)
        """
        if not answers:
            return 0.0, ""
        
        # Count occurrences
        counter = Counter(answers)
        majority_answer, majority_count = counter.most_common(1)[0]
        
        # Compute agreement rate
        agreement_rate = majority_count / len(answers)
        
        return agreement_rate, majority_answer
    
    def _format_baseline_prompt(self, question: str, benchmark_name: str) -> str:
        """Format baseline prompt (no metacognition)."""
        if benchmark_name == "gsm8k":
            return f"""Solve this math problem step by step.

Question: {question}

Provide your solution and final answer."""
        
        elif benchmark_name == "mmlu":
            return f"""Answer this multiple choice question.

{question}

Provide your answer (A, B, C, or D) and brief explanation."""
        
        elif benchmark_name == "hellaswag":
            return f"""Complete this scenario by selecting the most plausible continuation.

{question}

Provide your answer (0, 1, 2, or 3) and brief explanation."""
        
        elif benchmark_name == "mrben":
            return f"""Answer this multi-step reasoning question.

{question}

Provide your reasoning and final answer."""
        
        else:
            return f"""Answer this question.

{question}

Provide your answer and explanation."""
    
    def _format_metacognitive_prompt(self, question: str, benchmark_name: str) -> str:
        """Format metacognitive prompt (full 6-step process)."""
        if benchmark_name == "gsm8k":
            return f"""Solve this math problem using careful metacognitive reasoning.

Question: {question}

Follow these steps:
1. Clarify your understanding of what the problem is asking.
2. Make a preliminary solution to the problem.
3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
4. Once you have a solution, decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation of your reasoning.
6. Rate your overall confidence (0-100%) in this answer and explain why."""
        
        elif benchmark_name == "mmlu":
            return f"""Answer this multiple choice question using careful metacognitive reasoning.

{question}

Follow these steps:
1. Clarify your understanding of what the question is asking.
2. Make a preliminary analysis of each option.
3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
4. Once you have selected an option, decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation of your reasoning.
6. Rate your overall confidence (0-100%) in this answer and explain why."""
        
        elif benchmark_name == "hellaswag":
            return f"""Select the most plausible continuation using careful metacognitive reasoning.

{question}

Follow these steps:
1. Clarify your understanding of the scenario.
2. Make a preliminary analysis of each option.
3. Monitor your confidence in the selection. If you think it's wrong or has potential errors, pause and verify your work.
4. Once you have selected an option, decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation of your reasoning.
6. Rate your overall confidence (0-100%) in this answer and explain why."""
        
        elif benchmark_name == "mrben":
            return f"""Answer this multi-step reasoning question using careful metacognitive reasoning.

{question}

Follow these steps:
1. Clarify your understanding of what the question is asking.
2. Make a preliminary solution to the problem.
3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
4. Once you have a solution, decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation of your reasoning.
6. Rate your overall confidence (0-100%) in this answer and explain why."""
        
        else:
            return f"""Answer this question using careful metacognitive reasoning.

{question}

Follow these steps:
1. Clarify your understanding of what the question is asking.
2. Make a preliminary analysis.
3. Monitor your confidence. If you think it's wrong or has potential errors, pause and verify your work.
4. Decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation.
6. Rate your overall confidence (0-100%) and explain why."""
    
    def evaluate_single(self, sample: Dict, benchmark_name: str) -> Dict:
        """
        Evaluate a single sample using self-consistency based selective metacognition.
        
        Args:
            sample: The sample to evaluate
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary with evaluation results
        """
        question = sample.get("question", sample.get("input", ""))
        ground_truth = sample.get("answer", sample.get("target", ""))
        
        # Step 1: Generate multiple baseline answers
        baseline_answers = self.generate_multiple_answers(question, benchmark_name)
        
        # Step 2: Compute agreement
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
            metacog_response = self.model.generate(
                metacog_prompt,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=False
            )
            final_answer = self.extract_answer(metacog_response, benchmark_name)
            method_used = "metacognition"
            full_response = f"Baseline answers: {baseline_answers}\nAgreement: {agreement_rate:.2f}\nMetacognitive response: {metacog_response}"
        
        # Check correctness
        is_correct = self.check_answer(final_answer, ground_truth, benchmark_name)
        
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
        print(f"SELF-CONSISTENCY EVALUATION: {benchmark_name.upper()}")
        print(f"{'='*60}")
        print(f"N samples: {self.n_samples}")
        print(f"Agreement threshold: {self.agreement_threshold}")
        print(f"Temperature: {self.temperature}")
        
        # Load benchmark data
        samples = self.load_benchmark_data(benchmark_name, num_samples)
        print(f"Loaded {len(samples)} samples")
        
        # Evaluate each sample
        results = []
        for sample in tqdm(samples, desc=f"Evaluating {benchmark_name}"):
            result = self.evaluate_single(sample, benchmark_name)
            results.append(result)
        
        # Compute statistics
        correct_count = sum(1 for r in results if r["correct"])
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        baseline_count = sum(1 for r in results if r["method_used"] == "baseline")
        metacog_count = sum(1 for r in results if r["method_used"] == "metacognition")
        
        baseline_correct = sum(1 for r in results if r["method_used"] == "baseline" and r["correct"])
        metacog_correct = sum(1 for r in results if r["method_used"] == "metacognition" and r["correct"])
        
        baseline_accuracy = baseline_correct / baseline_count if baseline_count > 0 else 0
        metacog_accuracy = metacog_correct / metacog_count if metacog_count > 0 else 0
        
        avg_agreement = sum(r["agreement_rate"] for r in results) / total_count if total_count > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {benchmark_name.upper()}")
        print(f"{'='*60}")
        print(f"Overall accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        print(f"\nMethod distribution:")
        print(f"  Baseline: {baseline_count} ({baseline_count/total_count:.1%})")
        print(f"  Metacognition: {metacog_count} ({metacog_count/total_count:.1%})")
        print(f"\nAccuracy by method:")
        print(f"  Baseline: {baseline_accuracy:.2%} ({baseline_correct}/{baseline_count})")
        print(f"  Metacognition: {metacog_accuracy:.2%} ({metacog_correct}/{metacog_count})")
        print(f"\nAverage agreement rate: {avg_agreement:.2%}")
        print(f"{'='*60}\n")
        
        return {
            "benchmark": benchmark_name,
            "n_samples": self.n_samples,
            "agreement_threshold": self.agreement_threshold,
            "temperature": self.temperature,
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total_count,
            "baseline_count": baseline_count,
            "metacog_count": metacog_count,
            "baseline_accuracy": baseline_accuracy,
            "metacog_accuracy": metacog_accuracy,
            "avg_agreement": avg_agreement,
            "results": results
        }


def main():
    parser = argparse.ArgumentParser(description="Self-Consistency Based Selective Metacognition Evaluator")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Name of the model to evaluate")
    parser.add_argument("--benchmarks", type=str, default="gsm8k,mmlu",
                        help="Comma-separated list of benchmarks to evaluate")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate per benchmark (None for all)")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of samples for self-consistency check")
    parser.add_argument("--agreement_threshold", type=float, default=0.67,
                        help="Agreement threshold (0-1). If agreement >= threshold, use baseline.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling diverse answers")
    parser.add_argument("--output_dir", type=str, default="results/selfconsistency",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = SelfConsistencyEvaluator(
        model_name=args.model_name,
        n_samples=args.n_samples,
        agreement_threshold=args.agreement_threshold,
        temperature=args.temperature
    )
    
    # Evaluate each benchmark
    benchmarks = args.benchmarks.split(",")
    all_results = {}
    
    for benchmark in benchmarks:
        benchmark = benchmark.strip()
        result = evaluator.evaluate_benchmark(benchmark, args.num_samples)
        all_results[benchmark] = result
        
        # Save individual benchmark results
        output_file = os.path.join(args.output_dir, f"selfconsistency_{benchmark}_results.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    # Save summary
    summary = {
        "model": args.model_name,
        "n_samples": args.n_samples,
        "agreement_threshold": args.agreement_threshold,
        "temperature": args.temperature,
        "benchmarks": {
            name: {
                "accuracy": result["accuracy"],
                "baseline_count": result["baseline_count"],
                "metacog_count": result["metacog_count"],
                "baseline_accuracy": result["baseline_accuracy"],
                "metacog_accuracy": result["metacog_accuracy"],
                "avg_agreement": result["avg_agreement"]
            }
            for name, result in all_results.items()
        }
    }
    
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
