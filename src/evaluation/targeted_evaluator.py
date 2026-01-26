"""
Targeted Metacognitive Evaluator

This evaluator applies metacognition ONLY to questions that the baseline model got wrong.
This tests whether metacognition actually helps on difficult questions.

Workflow:
1. Load baseline results
2. Identify questions baseline got wrong
3. Apply metacognition to those specific questions
4. Compare: Does metacognition fix the errors?
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
from tqdm import tqdm
from src.evaluation.threshold_evaluator import ThresholdMetacognitiveEvaluator


class TargetedMetacognitiveEvaluator:
    """
    Evaluator that applies metacognition only to baseline errors.
    """
    
    def __init__(self, model_name, device='cuda', max_new_tokens=2048):
        self.evaluator = ThresholdMetacognitiveEvaluator(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens
        )
        self.model_name = model_name
    
    def load_baseline_results(self, baseline_file):
        """Load baseline results to identify errors"""
        with open(baseline_file, 'r') as f:
            data = json.load(f)
        return data
    
    def evaluate_errors_only(self, baseline_file, benchmark_name):
        """
        Evaluate only the questions that baseline got wrong.
        
        Args:
            baseline_file: Path to baseline results JSON
            benchmark_name: Name of benchmark (gsm8k, mmlu, etc.)
        
        Returns:
            Dictionary with results
        """
        # Load baseline results
        print(f"Loading baseline results from: {baseline_file}")
        baseline_data = self.load_baseline_results(baseline_file)
        
        # Find errors - handle both formats
        if 'results' in baseline_data:
            # Metacognitive evaluator format
            all_samples = baseline_data['results']
            error_samples = [s for s in all_samples if not s.get('correct', False)]
        elif 'predictions' in baseline_data:
            # Baseline evaluator format
            all_samples = baseline_data['predictions']
            error_samples = [s for s in all_samples if not s.get('is_correct', True)]
        else:
            raise ValueError("Baseline file doesn't have 'results' or 'predictions' field")
        
        print(f"Baseline accuracy: {baseline_data.get('accuracy', 'N/A')}%")
        print(f"Total samples: {len(all_samples)}")
        print(f"Baseline errors: {len(error_samples)}")
        print(f"\nApplying metacognition to {len(error_samples)} error cases...")
        
        # Evaluate error cases with metacognition
        results = []
        fixed_count = 0
        still_wrong_count = 0
        
        for sample in tqdm(error_samples, desc=f"Evaluating {benchmark_name} errors"):
            question = sample['question']
            ground_truth = sample.get('ground_truth', sample.get('target_answer', 'N/A'))
            
            # Generate answer with metacognition
            try:
                full_response, predicted_answer = self.evaluator.generate_answer(
                    question=question,
                    benchmark_name=benchmark_name
                )
                
                # Check if metacognition fixed the error
                correct = (predicted_answer == ground_truth)
                
                if correct:
                    fixed_count += 1
                else:
                    still_wrong_count += 1
                
                results.append({
                    'sample_id': sample.get('sample_id', sample.get('id', -1)),
                    'question': question,
                    'ground_truth': ground_truth,
                    'baseline_prediction': sample.get('predicted', sample.get('predicted_answer', 'N/A')),
                    'baseline_output': sample.get('model_output', sample.get('full_response', 'N/A')),
                    'metacog_prediction': predicted_answer,
                    'fixed': correct,
                    'full_response': full_response
                })
                
            except Exception as e:
                print(f"\nError on sample {sample.get('sample_id', '?')}: {e}")
                results.append({
                    'sample_id': sample.get('sample_id', sample.get('id', -1)),
                    'question': question,
                    'ground_truth': ground_truth,
                    'baseline_prediction': sample.get('predicted', sample.get('predicted_answer', 'N/A')),
                    'metacog_prediction': 'ERROR',
                    'fixed': False,
                    'error': str(e)
                })
                still_wrong_count += 1
        
        # Calculate statistics
        fix_rate = (fixed_count / len(error_samples) * 100) if error_samples else 0
        
        # Calculate what the new overall accuracy would be
        baseline_correct = baseline_data.get('correct', baseline_data.get('num_correct', 0))
        baseline_accuracy = baseline_data.get('accuracy', 0)
        total_samples = len(all_samples)
        new_correct = baseline_correct + fixed_count
        new_accuracy = (new_correct / total_samples * 100) if total_samples else 0
        
        return {
            'benchmark': benchmark_name,
            'model': self.model_name,
            'evaluation_type': 'targeted_metacognitive',
            'baseline_accuracy': baseline_accuracy,
            'baseline_errors': len(error_samples),
            'errors_fixed': fixed_count,
            'still_wrong': still_wrong_count,
            'fix_rate': fix_rate,
            'projected_new_accuracy': new_accuracy,
            'improvement': new_accuracy - baseline_data.get('accuracy', 0),
            'results': results
        }


def main():
    """Main evaluation function"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description='Apply metacognition only to baseline errors'
    )
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Model name or path')
    parser.add_argument('--baseline_file', type=str, required=True,
                       help='Path to baseline results JSON file')
    parser.add_argument('--benchmark', type=str, required=True,
                       help='Benchmark name (gsm8k, mmlu, hellaswag, mrben)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use')
    parser.add_argument('--max_new_tokens', type=str, default=2048,
                       help='Maximum new tokens to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    print(f"Loading model: {args.model_name}")
    evaluator = TargetedMetacognitiveEvaluator(
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )
    
    # Evaluate
    print(f"\n{'='*60}")
    print(f"TARGETED EVALUATION: {args.benchmark.upper()}")
    print('='*60)
    
    result = evaluator.evaluate_errors_only(
        baseline_file=args.baseline_file,
        benchmark_name=args.benchmark
    )
    
    # Save results
    output_file = os.path.join(
        args.output_dir, 
        f'targeted_{args.benchmark}_results.json'
    )
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.benchmark.upper()}")
    print('='*60)
    print(f"Baseline accuracy: {result['baseline_accuracy']:.2f}%")
    print(f"Baseline errors: {result['baseline_errors']}")
    print(f"Errors fixed by metacognition: {result['errors_fixed']}")
    print(f"Still wrong: {result['still_wrong']}")
    print(f"Fix rate: {result['fix_rate']:.2f}%")
    print(f"\nProjected new accuracy: {result['projected_new_accuracy']:.2f}%")
    print(f"Improvement: {result['improvement']:+.2f}%")
    print('='*60)
    print(f"\nResults saved to: {output_file}\n")


if __name__ == "__main__":
    main()
