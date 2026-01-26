"""
Threshold-Based Metacognitive Evaluator

This evaluator uses confidence thresholds instead of binary SIMPLE/COMPLEX classification.
The model generates an initial answer with confidence, then applies metacognition based on threshold.

Confidence levels:
- ≥ 90%: Direct answer (no metacognition)
- 70-89%: Light verification
- < 70%: Full metacognitive analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.evaluation.metacognitive_evaluator import MetacognitiveEvaluator
import torch

class ThresholdMetacognitiveEvaluator(MetacognitiveEvaluator):
    """
    Threshold-based metacognitive evaluator that applies different levels of
    metacognition based on model's self-reported confidence.
    """
    
    def __init__(self, model_name, device='cuda', max_new_tokens=2048):
        super().__init__(model_name, device, max_new_tokens)
        self.evaluation_type = "threshold_metacognitive"
    
    def _format_metacognitive_prompt(self, question, benchmark_name=None):
        """
        Format prompt with threshold-based metacognition.
        
        This overrides the parent class's _format_metacognitive_prompt to use
        confidence thresholds instead of SIMPLE/COMPLEX classification.
        
        Args:
            question: The question to answer
            benchmark_name: Name of the benchmark (gsm8k, mmlu, etc.)
        
        Returns:
            Formatted prompt string ready for tokenization
        """
        
        # Simple system message
        system_message = """You are a helpful assistant that solves problems carefully and thoughtfully."""
        
        # Benchmark-specific user message formatting
        if benchmark_name and benchmark_name.lower() == 'gsm8k':
            user_message = f"""Solve this problem using confidence-based reasoning:

1. Generate your initial answer and rate your confidence (0-100%)

2. Based on your confidence level:
   - If confidence ≥ 90%: Provide your answer directly and skip to step 3
   - If confidence < 90%: Follow the detailed metacognitive steps below:

   a. Clarify your understanding of what the problem is asking.
   
   b. Make a preliminary solution to the problem.
   
   c. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
   
   d. Once you have a solution, decide whether you need additional verification or if you're confident enough to finalize.
   
   e. Provide your final answer with a clear explanation of your reasoning.
   
   f. Rate your overall confidence (0-100%) in this answer and explain why.

3. Provide your final answer

Problem: {question}

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [just the number, nothing else]

Example: If the answer is 25 dollars, write:
Final Answer: 25"""
        
        elif benchmark_name and benchmark_name.lower() in ['mmlu', 'hellaswag', 'mrben']:
            user_message = f"""Answer this question using confidence-based reasoning:

1. Generate your initial answer and rate your confidence (0-100%)

2. Based on your confidence level:
   - If confidence ≥ 90%: Provide your answer directly and skip to step 3
   - If confidence < 90%: Follow the detailed metacognitive steps below:

   a. Clarify your understanding of what the question is asking.
   
   b. Make a preliminary analysis of each option.
   
   c. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
   
   d. Once you have selected an option, decide whether you need additional verification or if you're confident enough to finalize.
   
   e. Provide your final answer with a clear explanation of your reasoning.
   
   f. Rate your overall confidence (0-100%) in this answer and explain why.

3. Provide your final answer

{question}

IMPORTANT: You MUST end your response with exactly this format:
Final Answer: [just the letter A, B, C, or D]

Example: If you choose option B, write:
Final Answer: B"""
        
        else:
            # Generic format
            user_message = f"""Solve this problem using confidence-based reasoning:

1. Generate your initial answer and rate your confidence (0-100%)

2. Based on your confidence level:
   - If confidence ≥ 90%: Provide your answer directly and skip to step 3
   - If confidence < 90%: Follow the detailed metacognitive steps below:

   a. Clarify your understanding of what is being asked.
   
   b. Make a preliminary solution to the problem.
   
   c. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
   
   d. Once you have a solution, decide whether you need additional verification or if you're confident enough to finalize.
   
   e. Provide your final answer with a clear explanation of your reasoning.
   
   f. Rate your overall confidence (0-100%) in this answer and explain why.

3. Provide your final answer

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


def main():
    """Main evaluation function"""
    import argparse
    import json
    from datetime import datetime
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description='Evaluate model with threshold-based metacognition')
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--benchmarks', type=str, nargs='+', required=True, 
                       help='Benchmarks to evaluate (gsm8k, mmlu, hellaswag, mrben)')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--max_new_tokens', type=int, default=2048, 
                       help='Maximum new tokens to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    print(f"Loading model: {args.model_name}")
    evaluator = ThresholdMetacognitiveEvaluator(
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )
    
    # Evaluate each benchmark
    for benchmark in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"Evaluating on {benchmark.upper()}")
        print('='*60)
        
        result = evaluator.evaluate_benchmark(
            benchmark_name=benchmark,
            max_samples=args.num_samples
        )
        
        # Save results
        output_file = os.path.join(args.output_dir, f'threshold_{benchmark}_test_results.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {benchmark.upper()}")
        print('='*60)
        print(f"Accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
        print('='*60)
        print(f"\nResults saved to: {output_file}\n")


if __name__ == "__main__":
    main()
