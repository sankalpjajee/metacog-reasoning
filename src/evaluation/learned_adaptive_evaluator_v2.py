"""
Learned Adaptive Evaluator V2

Uses trained error prediction probe to route between baseline and metacognitive prompting.

Key differences from V1:
- Uses multi-layer hidden states
- Uses baseline confidence features
- Uses self-consistency as a feature (not for decision)
- Predicts errors (not uncertainty)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.benchmarks import load_benchmark
from training.train_error_predictor_v2 import MLPProbe, EnsembleProbe


class LearnedAdaptiveEvaluatorV2:
    def __init__(
        self,
        model_path: str,
        probe_path: str,
        probe_config_path: str,
        device: str = "cuda",
    ):
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
        
        # Load probe
        print(f"Loading probe: {probe_path}")
        with open(probe_config_path, 'r') as f:
            probe_config = json.load(f)
        
        if probe_config['use_ensemble']:
            self.probe = EnsembleProbe(num_probes=probe_config['num_probes'])
        else:
            self.probe = MLPProbe()
        
        self.probe.load_state_dict(torch.load(probe_path))
        self.probe = self.probe.to(device)
        self.probe.eval()
        
        # Layers to extract hidden states from
        self.layers = [8, 16, 24, 32]
    
    def extract_features(self, question: str) -> Dict[str, torch.Tensor]:
        """Extract all features needed for probe prediction."""
        # 1. Multi-layer hidden states
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = []
        for layer_idx in self.layers:
            layer_hidden = outputs.hidden_states[layer_idx][0, -1, :]
            hidden_states.append(layer_hidden)
        
        combined_hidden = torch.cat(hidden_states, dim=0)
        
        # 2. Baseline confidence (quick generation)
        baseline_answer, confidence_features = self._generate_with_confidence(question, temperature=0.7)
        
        # 3. Self-consistency (3 samples)
        agreement_rate = self._compute_agreement(question, n_samples=3)
        
        return {
            'hidden_states': combined_hidden,
            'confidence_features': torch.tensor([
                confidence_features['mean_max_prob'],
                confidence_features['min_max_prob'],
                confidence_features['mean_entropy'],
                confidence_features['mean_top1_top2_gap'],
            ], dtype=torch.float32).to(self.device),
            'agreement_rate': torch.tensor(agreement_rate, dtype=torch.float32).to(self.device),
            'baseline_answer': baseline_answer,
        }
    
    def predict_error_probability(self, question: str) -> float:
        """Predict probability that baseline will be wrong."""
        features = self.extract_features(question)
        
        with torch.no_grad():
            logit = self.probe(
                features['hidden_states'].unsqueeze(0),
                features['confidence_features'].unsqueeze(0),
                features['agreement_rate'].unsqueeze(0),
            )
            prob = torch.sigmoid(logit).item()
        
        return prob
    
    def _generate_with_confidence(self, question: str, temperature: float = 0.7):
        """Generate answer and extract confidence features."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions concisely and accurately."},
            {"role": "user", "content": f"{question}\n\nProvide your answer:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Extract confidence features
        all_logits = torch.stack(outputs.scores, dim=0)
        all_probs = torch.softmax(all_logits, dim=-1)
        
        max_probs = all_probs.max(dim=-1).values
        top2_probs = torch.topk(all_probs, k=2, dim=-1).values
        
        confidence_features = {
            'mean_max_prob': max_probs.mean().item(),
            'min_max_prob': max_probs.min().item(),
            'mean_entropy': -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1).mean().item(),
            'mean_top1_top2_gap': (top2_probs[:, :, 0] - top2_probs[:, :, 1]).mean().item(),
        }
        
        return answer, confidence_features
    
    def _compute_agreement(self, question: str, n_samples: int = 3) -> float:
        """Compute self-consistency agreement rate."""
        answers = []
        for _ in range(n_samples):
            answer, _ = self._generate_with_confidence(question, temperature=0.7)
            answers.append(self._extract_final_answer(answer))
        
        if not answers or all(a == "" for a in answers):
            return 0.0
        
        from collections import Counter
        answer_counts = Counter(answers)
        most_common_count = answer_counts.most_common(1)[0][1]
        return most_common_count / len(answers)
    
    def generate_answer(self, question: str, use_metacog: bool, benchmark_name: str) -> str:
        """Generate answer using baseline or metacognitive prompt."""
        if use_metacog:
            return self._generate_metacognitive(question, benchmark_name)
        else:
            answer, _ = self._generate_with_confidence(question, temperature=0.7)
            return answer
    
    def _generate_metacognitive(self, question: str, benchmark_name: str) -> str:
        """Generate answer using metacognitive prompt."""
        # Use the 6-step metacognitive framework
        metacog_prompt = """Before answering, follow these steps:

1. Clarify your understanding of what the question is asking.

2. Make a preliminary analysis of each option.

3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.

4. Once you have selected an option, decide whether you need additional verification or if you're confident enough to finalize.

5. Provide your final answer with a clear explanation of your reasoning.

6. Rate your overall confidence (0-100%) in this answer and explain why.

"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Think carefully and verify your reasoning."},
            {"role": "user", "content": f"{metacog_prompt}\n{question}\n\nProvide your answer:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        return answer
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract final answer from generated text."""
        text = text.strip()
        
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].strip()
            answer = answer.split("\n")[0].strip()
            return answer
        
        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip()
            answer = answer.split("\n")[0].strip()
            return answer
        
        return text.split("\n")[0].strip()
    
    def _check_answer(self, predicted: str, ground_truth: str, benchmark_name: str) -> bool:
        """Check if predicted answer matches ground truth."""
        pred_norm = predicted.strip().lower()
        gt_norm = ground_truth.strip().lower()
        
        if benchmark_name.lower() in ['mmlu', 'hellaswag']:
            return pred_norm.upper() == gt_norm.upper()
        
        if pred_norm == gt_norm:
            return True
        
        try:
            pred_num = float(pred_norm.replace(',', ''))
            gt_num = float(gt_norm.replace(',', ''))
            return abs(pred_num - gt_num) < 1e-6
        except:
            pass
        
        return pred_norm in gt_norm or gt_norm in pred_norm
    
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        num_samples: int,
        error_threshold: float = 0.5,
    ) -> Dict:
        """Evaluate using learned adaptive approach."""
        print(f"\n{'='*60}")
        print(f"LEARNED ADAPTIVE EVALUATION V2: {benchmark_name.upper()}")
        print(f"{'='*60}")
        print(f"Error threshold: {error_threshold}")
        print(f"Num samples: {num_samples}")
        
        # Load benchmark
        samples = load_benchmark(benchmark_name, split="test")
        if len(samples) > num_samples:
            samples = samples[:num_samples]
        
        print(f"Loaded {len(samples)} samples\n")
        
        # Evaluate
        results = []
        for sample in tqdm(samples, desc=f"Evaluating {benchmark_name}"):
            question = sample['question']
            ground_truth = sample['answer']
            
            # Predict error probability
            error_prob = self.predict_error_probability(question)
            use_metacog = error_prob > error_threshold
            
            # Generate answer
            answer = self.generate_answer(question, use_metacog, benchmark_name)
            final_answer = self._extract_final_answer(answer)
            
            # Check correctness
            is_correct = self._check_answer(final_answer, ground_truth, benchmark_name)
            
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': final_answer,
                'is_correct': is_correct,
                'error_prob': error_prob,
                'used_metacog': use_metacog,
            })
        
        # Compute statistics
        num_correct = sum(1 for r in results if r['is_correct'])
        num_metacog = sum(1 for r in results if r['used_metacog'])
        num_baseline = len(results) - num_metacog
        
        baseline_correct = sum(1 for r in results if not r['used_metacog'] and r['is_correct'])
        metacog_correct = sum(1 for r in results if r['used_metacog'] and r['is_correct'])
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {benchmark_name.upper()}")
        print(f"{'='*60}")
        print(f"Overall accuracy: {num_correct/len(results)*100:.2f}% ({num_correct}/{len(results)})")
        print(f"\nMethod distribution:")
        print(f"  Baseline: {num_baseline} ({num_baseline/len(results)*100:.1f}%)")
        print(f"  Metacognition: {num_metacog} ({num_metacog/len(results)*100:.1f}%)")
        print(f"\nAccuracy by method:")
        if num_baseline > 0:
            print(f"  Baseline: {baseline_correct/num_baseline*100:.2f}% ({baseline_correct}/{num_baseline})")
        if num_metacog > 0:
            print(f"  Metacognition: {metacog_correct/num_metacog*100:.2f}% ({metacog_correct}/{num_metacog})")
        print(f"{'='*60}\n")
        
        return {
            'results': results,
            'accuracy': num_correct / len(results),
            'num_baseline': num_baseline,
            'num_metacog': num_metacog,
            'baseline_accuracy': baseline_correct / num_baseline if num_baseline > 0 else 0.0,
            'metacog_accuracy': metacog_correct / num_metacog if num_metacog > 0 else 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate using learned adaptive approach V2")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--probe_config", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "hellaswag"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--error_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="results/learned_adaptive_v2")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = LearnedAdaptiveEvaluatorV2(
        model_path=args.model_path,
        probe_path=args.probe_path,
        probe_config_path=args.probe_config,
    )
    
    # Evaluate each benchmark
    os.makedirs(args.output_dir, exist_ok=True)
    
    for benchmark in args.benchmarks:
        result = evaluator.evaluate_benchmark(
            benchmark_name=benchmark,
            num_samples=args.num_samples,
            error_threshold=args.error_threshold,
        )
        
        # Save results
        output_path = os.path.join(args.output_dir, f"learned_adaptive_v2_{benchmark}_results.json")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
