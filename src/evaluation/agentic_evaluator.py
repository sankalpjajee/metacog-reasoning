"""Agentic evaluator with uncertainty-driven multi-hop reasoning."""
import json
import os
from typing import List, Dict, Optional, Tuple
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hotpotqa_loader import load_hotpotqa
from .question_decomposer import QuestionDecomposer, SimpleDecomposer
from .benchmarks import BenchmarkSample


class AgenticEvaluator:
    """Evaluator for agentic multi-hop reasoning with adaptive strategy selection."""
    
    def __init__(self,
                 model_path: str,
                 probe_path: Optional[str] = None,
                 confidence_threshold: float = 0.7,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the agentic evaluator.
        
        Args:
            model_path: Path to the base language model
            probe_path: Path to the trained confidence probe (optional)
            confidence_threshold: Threshold for using metacognitive prompting
            device: Device to run on
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
        # Load confidence probe if provided
        self.probe = None
        if probe_path and os.path.exists(probe_path):
            print(f"Loading confidence probe from {probe_path}...")
            self.probe = self._load_probe(probe_path)
        
        # Initialize question decomposer
        self.decomposer = SimpleDecomposer()  # Use simple decomposer for now
        # For LLM-based decomposition: self.decomposer = QuestionDecomposer(self.model, self.tokenizer)
    
    def _load_probe(self, probe_path: str):
        """Load the trained confidence probe."""
        # TODO: Implement probe loading
        # This should load the 2-layer MLP trained in train_probe.py
        checkpoint = torch.load(probe_path, map_location=self.device)
        probe = checkpoint['model']  # Assuming model is saved in checkpoint
        probe.eval()
        return probe
    
    def evaluate_hotpotqa(self, 
                          num_samples: int = 1000,
                          split: str = "validation",
                          output_dir: str = "results/agentic") -> Dict:
        """Evaluate on HotPotQA with agentic multi-hop reasoning.
        
        Args:
            num_samples: Number of samples to evaluate
            split: Dataset split to use
            output_dir: Directory to save results
        
        Returns:
            Dictionary with evaluation results
        """
        # Load HotPotQA
        print(f"Loading HotPotQA {split} split...")
        samples = load_hotpotqa(split=split, setting="distractor", max_samples=num_samples)
        
        results = []
        correct = 0
        total_hops = 0
        metacog_hops = 0
        
        print(f"\nEvaluating {len(samples)} samples...")
        for sample in tqdm(samples):
            result = self.evaluate_sample(sample)
            results.append(result)
            
            if result['is_correct']:
                correct += 1
            
            total_hops += result['num_hops']
            metacog_hops += result['num_metacog_hops']
        
        # Calculate metrics
        accuracy = correct / len(samples) if samples else 0
        avg_hops = total_hops / len(samples) if samples else 0
        avg_metacog_hops = metacog_hops / len(samples) if samples else 0
        metacog_ratio = metacog_hops / total_hops if total_hops > 0 else 0
        
        summary = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(samples),
            'avg_hops': avg_hops,
            'avg_metacog_hops': avg_metacog_hops,
            'metacog_ratio': metacog_ratio,
        }
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'agentic_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(os.path.join(output_dir, 'agentic_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("AGENTIC EVALUATION RESULTS: HOTPOTQA")
        print("="*60)
        print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{len(samples)})")
        print(f"Average hops per question: {avg_hops:.2f}")
        print(f"Average metacognitive hops: {avg_metacog_hops:.2f}")
        print(f"Metacognition usage: {metacog_ratio*100:.1f}%")
        print("="*60)
        
        return summary
    
    def evaluate_sample(self, sample: BenchmarkSample) -> Dict:
        """Evaluate a single sample with multi-hop reasoning.
        
        Args:
            sample: BenchmarkSample to evaluate
        
        Returns:
            Dictionary with evaluation results
        """
        # Decompose question into hops
        hops = self.decomposer.decompose(sample.question)
        
        # Answer each hop sequentially
        previous_answers = []
        hop_results = []
        
        for i, hop in enumerate(hops):
            # Format hop with context from previous hops
            hop_with_context = self.decomposer.format_hop_with_context(
                hop, previous_answers
            )
            
            # Predict confidence for this hop
            use_metacog = self._should_use_metacognition(hop_with_context)
            
            # Generate answer
            if use_metacog:
                answer = self._answer_with_metacognition(hop_with_context)
            else:
                answer = self._answer_with_baseline(hop_with_context)
            
            # Store hop result
            hop_results.append({
                'hop': hop,
                'answer': answer,
                'used_metacognition': use_metacog
            })
            
            previous_answers.append({
                'hop': hop,
                'answer': answer
            })
        
        # Final answer is from the last hop
        final_answer = hop_results[-1]['answer'] if hop_results else ""
        
        # Check correctness
        is_correct = self._check_answer(final_answer, sample.answer)
        
        return {
            'id': sample.id,
            'question': sample.question,
            'ground_truth': sample.answer,
            'predicted_answer': final_answer,
            'is_correct': is_correct,
            'num_hops': len(hops),
            'num_metacog_hops': sum(1 for h in hop_results if h['used_metacognition']),
            'hop_results': hop_results
        }
    
    def _should_use_metacognition(self, question: str) -> bool:
        """Decide whether to use metacognitive prompting for this hop.
        
        Args:
            question: The hop question
        
        Returns:
            True if should use metacognition, False otherwise
        """
        if self.probe is None:
            # Without probe, use heuristic: use metacognition 30% of the time
            # In practice, this would be based on question complexity
            return False
        
        # Get hidden state for the question
        hidden_state = self._get_hidden_state(question)
        
        # Predict confidence
        with torch.no_grad():
            confidence_score = self.probe(hidden_state).item()
        
        # Use metacognition if confidence is low
        return confidence_score < self.confidence_threshold
    
    def _get_hidden_state(self, text: str) -> torch.Tensor:
        """Get the hidden state for a text input.
        
        Args:
            text: Input text
        
        Returns:
            Hidden state tensor
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get last layer hidden state at the last token position
            hidden_state = outputs.hidden_states[-1][0, -1, :]
        
        return hidden_state
    
    def _answer_with_baseline(self, question: str) -> str:
        """Answer with baseline prompting.
        
        Args:
            question: The question to answer
        
        Returns:
            The answer
        """
        prompt = f"""Answer the following question concisely.

{question}

Answer:"""
        
        return self._generate_answer(prompt)
    
    def _answer_with_metacognition(self, question: str) -> str:
        """Answer with metacognitive prompting.
        
        Args:
            question: The question to answer
        
        Returns:
            The answer
        """
        prompt = f"""This question requires careful reasoning. Follow these steps:

1. Clarify your understanding of what is being asked.
2. Consider what information you need to answer this.
3. Monitor your confidence as you reason.
4. Verify your answer before finalizing.

{question}

Work through this carefully and provide your final answer:"""
        
        return self._generate_answer(prompt)
    
    def _generate_answer(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate an answer using the model.
        
        Args:
            prompt: The prompt
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated answer
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer from response (remove prompt)
        answer = response[len(prompt):].strip()
        
        # Extract just the answer (before any explanation)
        answer = answer.split('\n')[0].strip()
        
        return answer
    
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
        
        Returns:
            True if correct, False otherwise
        """
        # Normalize answers
        pred_norm = predicted.lower().strip()
        gt_norm = ground_truth.lower().strip()
        
        # Exact match
        if pred_norm == gt_norm:
            return True
        
        # Check if ground truth is contained in prediction
        if gt_norm in pred_norm:
            return True
        
        # Check if prediction is contained in ground truth
        if pred_norm in gt_norm:
            return True
        
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate agentic multi-hop reasoning")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Path to the base model")
    parser.add_argument("--probe_path", type=str, default=None,
                        help="Path to the trained confidence probe")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Confidence threshold for using metacognition")
    parser.add_argument("--output_dir", type=str, default="results/agentic",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    evaluator = AgenticEvaluator(
        model_path=args.model_path,
        probe_path=args.probe_path,
        confidence_threshold=args.confidence_threshold
    )
    
    evaluator.evaluate_hotpotqa(
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
