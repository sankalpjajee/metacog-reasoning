"""
ACC-Inspired Adaptive Evaluator (V4.1)

V4.1 removes compressed hidden states - uses only dynamic features + early entropy.
No PCA needed!

Uses the trained ACC probe to route between baseline and metacognition
based on predicted UTILITY GAIN (not just difficulty).

Key innovation: Only applies metacognition when expected improvement > cost.
"""

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ACCProbe(nn.Module):
    """ACC-Inspired probe (must match training architecture)."""
    def __init__(
        self,
        input_dim: int = 11,  # V4.1: 9 dynamic + 2 entropy
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 64)
        
        self.wrong_head = nn.Linear(64, 1)
        self.conflict_head = nn.Linear(64, 1)
        self.utility_head = nn.Linear(64, 1)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        wrong_logits = self.wrong_head(x).squeeze(-1)
        conflict_score = torch.sigmoid(self.conflict_head(x)).squeeze(-1)
        utility_score = torch.tanh(self.utility_head(x)).squeeze(-1)
        
        return wrong_logits, conflict_score, utility_score


class EnsembleACCProbe(nn.Module):
    """Ensemble of ACC probes."""
    def __init__(self, num_probes: int = 3, **probe_kwargs):
        super().__init__()
        self.probes = nn.ModuleList([ACCProbe(**probe_kwargs) for _ in range(num_probes)])
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wrong_preds = []
        conflict_preds = []
        utility_preds = []
        
        for probe in self.probes:
            wrong, conflict, utility = probe(features)
            wrong_preds.append(wrong)
            conflict_preds.append(conflict)
            utility_preds.append(utility)
        
        ensemble_wrong = torch.stack(wrong_preds, dim=0).mean(dim=0)
        ensemble_conflict = torch.stack(conflict_preds, dim=0).mean(dim=0)
        ensemble_utility = torch.stack(utility_preds, dim=0).mean(dim=0)
        
        return ensemble_wrong, ensemble_conflict, ensemble_utility


class ACCAdaptiveEvaluator:
    """
    Evaluator that uses ACC probe for value-aware routing.
    
    Routes to metacognition only when predicted utility > threshold.
    """
    
    def __init__(
        self,
        model_path: str,
        probe_path: str,
        probe_config_path: str,
        utility_threshold: float = 0.0,
        layers: List[int] = [8, 16, 24, 32],
    ):
        self.utility_threshold = utility_threshold
        self.layers = layers
        
        # Load LLM
        print(f"Loading model: {model_path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        
        # V4.1: No PCA needed!
        
        # Load probe config
        with open(probe_config_path, 'r') as f:
            config = json.load(f)
        
        # Load probe
        print(f"Loading probe: {probe_path}")
        if config.get('use_ensemble', False):
            self.probe = EnsembleACCProbe(
                num_probes=config['num_probes'],
                input_dim=config['input_dim'],
            )
        else:
            self.probe = ACCProbe(input_dim=config['input_dim'])
        
        self.probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
        self.probe.eval()
        self.probe.to(self.device)
    
    def extract_features(self, question: str) -> torch.Tensor:
        """Extract ACC features for a question."""
        # 1. Extract multi-layer hidden states
        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = {}
        for layer_idx in self.layers:
            if layer_idx < len(outputs.hidden_states):
                h = outputs.hidden_states[layer_idx][0, -1, :].cpu()
                hidden_states[f"layer_{layer_idx}"] = h
        
        # V4.1: Only dynamic features + early entropy (no compressed hidden)
        dynamic_features = self._compute_dynamic_features(hidden_states)
        entropy_features = self._compute_early_entropy(question)
        
        # Combine features (11 dims total)
        features = torch.cat([
            dynamic_features,   # 9 dims
            entropy_features,   # 2 dims
        ])
        
        return features
    
    def _compute_dynamic_features(self, hidden_states: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute dynamic features across layers."""
        layers = sorted([int(k.split("_")[1]) for k in hidden_states.keys()])
        features = []
        
        for i in range(1, len(layers)):
            h_prev = hidden_states[f"layer_{layers[i-1]}"]
            h_curr = hidden_states[f"layer_{layers[i]}"]
            
            # Cosine drift
            cosine_sim = F.cosine_similarity(h_prev.unsqueeze(0), h_curr.unsqueeze(0))
            cosine_drift = 1.0 - cosine_sim.item()
            features.append(cosine_drift)
            
            # Norm ratio
            norm_prev = torch.norm(h_prev).item()
            norm_curr = torch.norm(h_curr).item()
            norm_ratio = norm_curr / (norm_prev + 1e-8)
            features.append(norm_ratio)
            
            # Residual change
            residual = h_curr - h_prev
            residual_norm = torch.norm(residual).item()
            residual_change = residual_norm / (norm_prev + 1e-8)
            features.append(residual_change)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _compute_early_entropy(self, question: str) -> torch.Tensor:
        """Compute early branching entropy (2 tokens)."""
        prompt = self._format_baseline_prompt(question, "generic")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )
        
        entropies = []
        for logits in outputs.scores[:2]:
            probs = torch.softmax(logits[0], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            entropies.append(entropy)
        
        if len(entropies) == 0:
            return torch.tensor([0.0, 0.0], dtype=torch.float32)
        
        return torch.tensor([np.mean(entropies), np.max(entropies)], dtype=torch.float32)
    
    def predict_utility(self, question: str) -> Tuple[float, float, float]:
        """Predict utility gain for a question."""
        features = self.extract_features(question)
        features = features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            wrong_logits, conflict_score, utility_score = self.probe(features)
        
        p_wrong = torch.sigmoid(wrong_logits).item()
        conflict = conflict_score.item()
        utility = utility_score.item()
        
        return p_wrong, conflict, utility
    
    def _format_baseline_prompt(self, question: str, benchmark: str) -> str:
        """Format baseline prompt."""
        if benchmark == "gsm8k":
            return f"""Solve this math problem step by step.

Question: {question}

Provide your solution and end with "Final Answer: [number]"."""
        
        elif benchmark in ["mmlu", "hellaswag"]:
            return f"""{question}

Answer with the letter only (A, B, C, or D).
Final Answer: [letter]"""
        
        return f"""{question}

Provide a clear answer.
Final Answer: [answer]"""
    
    def _format_metacog_prompt(self, question: str, benchmark: str) -> str:
        """Format metacognitive prompt (6-step)."""
        if benchmark == "gsm8k":
            return f"""You are solving a math problem. Use metacognitive reasoning:

1. Clarify your understanding of what the question is asking.
2. Make a preliminary solution attempt.
3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
4. Once you have a solution, decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation of your reasoning.
6. Rate your overall confidence (0-100%) in this answer and explain why.

Question: {question}

Work through this systematically, then provide:
Final Answer: [number]"""
        
        else:
            return f"""You are answering a multiple choice question. Use metacognitive reasoning:

1. Clarify your understanding of what the question is asking.
2. Make a preliminary analysis of each option.
3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
4. Once you have selected an option, decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation of your reasoning.
6. Rate your overall confidence (0-100%) in this answer and explain why.

{question}

Work through this systematically, then provide:
Final Answer: [letter A, B, C, or D]"""
    
    def _generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate answer from model."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
    
    def _extract_answer(self, response: str, benchmark: str) -> str:
        """Extract answer from response."""
        response = response.strip()
        
        # Look for "Final Answer: X" pattern
        final_match = re.search(r'Final Answer:\s*([^\n]+)', response, re.IGNORECASE)
        if final_match:
            answer = final_match.group(1).strip()
            answer = re.sub(r'[^\w\d\.\-]', '', answer)
            return answer
        
        if benchmark == "gsm8k":
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                return numbers[-1]
        
        elif benchmark in ["mmlu", "hellaswag"]:
            letters = re.findall(r'\b([A-D])\b', response.upper())
            if letters:
                return letters[-1]
        
        return ""
    
    def _check_answer(self, predicted: str, ground_truth: str, benchmark: str) -> bool:
        """Check if answer is correct."""
        pred = predicted.strip().upper()
        gt = ground_truth.strip().upper()
        
        if benchmark == "gsm8k":
            try:
                pred_num = float(re.sub(r'[^\d\.\-]', '', pred))
                gt_num = float(re.sub(r'[^\d\.\-]', '', gt))
                return abs(pred_num - gt_num) < 0.01
            except:
                return pred == gt
        
        else:
            return pred == gt
    
    def evaluate_sample(
        self,
        question: str,
        ground_truth: str,
        benchmark: str,
    ) -> Dict:
        """Evaluate a single sample with utility-based routing."""
        
        # Predict utility
        p_wrong, conflict, utility = self.predict_utility(question)
        
        # Route based on utility
        use_metacog = utility > self.utility_threshold
        
        if use_metacog:
            prompt = self._format_metacog_prompt(question, benchmark)
            max_tokens = 1024
        else:
            prompt = self._format_baseline_prompt(question, benchmark)
            max_tokens = 512
        
        # Generate answer
        response = self._generate_answer(prompt, max_tokens)
        answer = self._extract_answer(response, benchmark)
        is_correct = self._check_answer(answer, ground_truth, benchmark)
        
        return {
            'question': question[:200],
            'ground_truth': ground_truth,
            'predicted_answer': answer,
            'is_correct': is_correct,
            'method': 'metacog' if use_metacog else 'baseline',
            'p_wrong': p_wrong,
            'conflict': conflict,
            'utility': utility,
        }
    
    def evaluate_benchmark(
        self,
        benchmark: str,
        num_samples: int = 1000,
    ) -> Dict:
        """Evaluate on a benchmark."""
        from datasets import load_dataset
        import random
        
        # Load samples
        print(f"\nLoading {benchmark}...")
        samples = []
        
        if benchmark == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="test")
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            for idx in indices:
                item = dataset[idx]
                answer = item['answer'].split('####')[-1].strip()
                samples.append({
                    'question': item['question'],
                    'answer': answer,
                })
        
        elif benchmark == "mmlu":
            subjects = ["abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry"]
            samples_per_subject = num_samples // len(subjects)
            for subject in subjects:
                try:
                    dataset = load_dataset("cais/mmlu", subject, split="test")
                    indices = random.sample(range(len(dataset)), min(samples_per_subject, len(dataset)))
                    for idx in indices:
                        item = dataset[idx]
                        q = item['question']
                        options = item['choices']
                        formatted = f"{q}\n\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}"
                        answer = ['A', 'B', 'C', 'D'][item['answer']]
                        samples.append({'question': formatted, 'answer': answer})
                except:
                    continue
        
        elif benchmark == "hellaswag":
            dataset = load_dataset("Rowan/hellaswag", split="validation")
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            for idx in indices:
                item = dataset[idx]
                ctx = item['ctx']
                endings = item['endings']
                formatted = f"{ctx}\n\nA. {endings[0]}\nB. {endings[1]}\nC. {endings[2]}\nD. {endings[3]}"
                answer = ['A', 'B', 'C', 'D'][int(item['label'])]
                samples.append({'question': formatted, 'answer': answer})
        
        print(f"Loaded {len(samples)} samples")
        
        # Evaluate
        results = []
        for sample in tqdm(samples, desc=f"Evaluating {benchmark}"):
            result = self.evaluate_sample(sample['question'], sample['answer'], benchmark)
            results.append(result)
        
        # Compute statistics
        correct = sum(1 for r in results if r['is_correct'])
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        baseline_results = [r for r in results if r['method'] == 'baseline']
        metacog_results = [r for r in results if r['method'] == 'metacog']
        
        baseline_correct = sum(1 for r in baseline_results if r['is_correct'])
        metacog_correct = sum(1 for r in metacog_results if r['is_correct'])
        
        baseline_acc = baseline_correct / len(baseline_results) if baseline_results else 0
        metacog_acc = metacog_correct / len(metacog_results) if metacog_results else 0
        
        summary = {
            'benchmark': benchmark,
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'baseline_count': len(baseline_results),
            'baseline_correct': baseline_correct,
            'baseline_accuracy': baseline_acc,
            'metacog_count': len(metacog_results),
            'metacog_correct': metacog_correct,
            'metacog_accuracy': metacog_acc,
            'utility_threshold': self.utility_threshold,
        }
        
        return {'summary': summary, 'results': results}


def main():
    parser = argparse.ArgumentParser(description="ACC-Inspired Adaptive Evaluator (V4)")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--probe_path", type=str, required=True)
    # V4.1: No PCA needed!
    parser.add_argument("--probe_config_path", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "hellaswag"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--utility_threshold", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = ACCAdaptiveEvaluator(
        model_path=args.model_path,
        probe_path=args.probe_path,
        probe_config_path=args.probe_config_path,
        utility_threshold=args.utility_threshold,
    )
    
    # Evaluate each benchmark
    all_summaries = []
    
    for benchmark in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"ACC-ADAPTIVE V4 EVALUATION: {benchmark.upper()}")
        print(f"Utility threshold: {args.utility_threshold}")
        print(f"{'='*60}")
        
        result = evaluator.evaluate_benchmark(benchmark, args.num_samples)
        
        # Print summary
        s = result['summary']
        print(f"\n{'='*60}")
        print(f"RESULTS: {benchmark.upper()}")
        print(f"{'='*60}")
        print(f"Overall accuracy: {s['accuracy']*100:.2f}% ({s['correct']}/{s['total']})")
        print(f"\nMethod distribution:")
        print(f"  Baseline: {s['baseline_count']} ({s['baseline_count']/s['total']*100:.1f}%)")
        print(f"  Metacognition: {s['metacog_count']} ({s['metacog_count']/s['total']*100:.1f}%)")
        print(f"\nAccuracy by method:")
        print(f"  Baseline: {s['baseline_accuracy']*100:.2f}% ({s['baseline_correct']}/{s['baseline_count']})")
        print(f"  Metacognition: {s['metacog_accuracy']*100:.2f}% ({s['metacog_correct']}/{s['metacog_count']})")
        print(f"{'='*60}")
        
        all_summaries.append(s)
        
        # Save results
        result_path = os.path.join(args.output_dir, f"acc_v4_{benchmark}_results.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {result_path}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    for s in all_summaries:
        print(f"\n{s['benchmark'].upper()}:")
        print(f"  Accuracy: {s['accuracy']*100:.2f}%")
        print(f"  Baseline used: {s['baseline_count']} ({s['baseline_accuracy']*100:.1f}% acc)")
        print(f"  Metacog used: {s['metacog_count']} ({s['metacog_accuracy']*100:.1f}% acc)")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
