"""
FOK-Inspired Adaptive Evaluator (V5)

Uses the trained V5 FOK probe to route between baseline and metacognition.
Extracts all 45 FOK features at inference time for each question.

Key differences from V4.1:
  - Extracts 45 features (vs 11 in V4.1)
  - Dual-stream: runs both baseline and metacog feature extraction
  - FOK features: answer-position gap, logit lens, temporal dynamics
  - Confidence calibration via KNN on training data
  - New FOKProbe architecture (3-layer MLP with BatchNorm)
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
# Model Architecture (must match training)
# ============================================================

class FOKProbe(nn.Module):
    """FOK-Inspired probe (must match training architecture)."""
    def __init__(
        self,
        input_dim: int = 45,
        hidden_dim: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(dropout)

        self.wrong_head = nn.Linear(32, 1)
        self.conflict_head = nn.Linear(32, 1)
        self.utility_head = nn.Linear(32, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.fc1(features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        wrong_logits = self.wrong_head(x).squeeze(-1)
        conflict_score = torch.sigmoid(self.conflict_head(x)).squeeze(-1)
        utility_score = self.utility_head(x).squeeze(-1)

        return wrong_logits, conflict_score, utility_score


class EnsembleFOKProbe(nn.Module):
    """Ensemble of FOK probes."""
    def __init__(self, num_probes: int = 3, **probe_kwargs):
        super().__init__()
        self.probes = nn.ModuleList([FOKProbe(**probe_kwargs) for _ in range(num_probes)])

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wrong_preds, conflict_preds, utility_preds = [], [], []
        for probe in self.probes:
            w, c, u = probe(features)
            wrong_preds.append(w)
            conflict_preds.append(c)
            utility_preds.append(u)
        return (
            torch.stack(wrong_preds, dim=0).mean(dim=0),
            torch.stack(conflict_preds, dim=0).mean(dim=0),
            torch.stack(utility_preds, dim=0).mean(dim=0),
        )


# ============================================================
# Feature Extraction Functions (same as data generation)
# ============================================================

def compute_dynamic_features(hidden_states: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute dynamic features across layers (9 dims)."""
    layers = sorted([int(k.split("_")[1]) for k in hidden_states.keys()])
    features = []
    for i in range(1, len(layers)):
        h_prev = hidden_states[f"layer_{layers[i-1]}"]
        h_curr = hidden_states[f"layer_{layers[i]}"]

        cosine_sim = F.cosine_similarity(h_prev.unsqueeze(0), h_curr.unsqueeze(0))
        features.append(1.0 - cosine_sim.item())

        norm_prev = torch.norm(h_prev).item()
        norm_curr = torch.norm(h_curr).item()
        features.append(norm_curr / (norm_prev + 1e-8))

        residual = h_curr - h_prev
        features.append(torch.norm(residual).item() / (norm_prev + 1e-8))

    return torch.tensor(features, dtype=torch.float32)


def compute_early_entropy(logits_list: List[torch.Tensor], num_tokens: int = 2) -> torch.Tensor:
    """Compute early entropy from first N token logits (2 dims)."""
    entropies = []
    for logits in logits_list[:num_tokens]:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)
    if len(entropies) == 0:
        return torch.tensor([0.0, 0.0], dtype=torch.float32)
    return torch.tensor([np.mean(entropies), np.max(entropies)], dtype=torch.float32)


def compute_answer_position_gap(logits: torch.Tensor) -> float:
    """Top-1 vs Top-2 probability gap at answer position (1 dim)."""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, _ = torch.sort(probs, descending=True)
    return (sorted_probs[0] - sorted_probs[1]).item()


def compute_divergence_speed(logits_list: List[torch.Tensor]) -> float:
    """How quickly metacog resolves uncertainty (1 dim)."""
    if len(logits_list) < 2:
        return 0.0
    entropies = []
    for logits in logits_list[:2]:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)
    return entropies[0] - entropies[1]


def compute_layer_entropy_logit_lens(
    model, hidden_states: Dict[str, torch.Tensor], layers: List[int] = [8, 16, 24, 32]
) -> torch.Tensor:
    """Layer entropy via logit lens (4 dims)."""
    lm_head = model.lm_head
    entropies = []
    for layer_idx in layers:
        key = f"layer_{layer_idx}"
        if key not in hidden_states:
            entropies.append(0.0)
            continue
        h = hidden_states[key].to(lm_head.weight.device).to(lm_head.weight.dtype)
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            h = model.model.norm(h.unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            logits = lm_head(h)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            entropies.append(entropy)
    return torch.tensor(entropies, dtype=torch.float32)


def compute_cross_comparison(
    baseline_first_token: int, metacog_first_token: int,
    baseline_gap: float, metacog_gap: float,
) -> torch.Tensor:
    """Token divergence + answer gap difference (2 dims)."""
    token_divergence = 1.0 if baseline_first_token != metacog_first_token else 0.0
    gap_difference = metacog_gap - baseline_gap
    return torch.tensor([token_divergence, gap_difference], dtype=torch.float32)


def compute_early_answer_agreement(
    baseline_tokens: List[int], metacog_tokens: List[int],
    baseline_logits: List[torch.Tensor], metacog_logits: List[torch.Tensor],
    num_tokens: int = 5,
) -> torch.Tensor:
    """Token agreement, KL divergence, top-k overlap (3 dims)."""
    n = min(num_tokens, len(baseline_tokens), len(metacog_tokens),
            len(baseline_logits), len(metacog_logits))
    if n == 0:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    matches = sum(1 for i in range(n) if baseline_tokens[i] == metacog_tokens[i])
    agreement_rate = matches / n

    kl_divs, top_k_overlaps = [], []
    for i in range(n):
        b_probs = torch.softmax(baseline_logits[i], dim=-1)
        m_probs = torch.softmax(metacog_logits[i], dim=-1)
        kl = F.kl_div(torch.log(m_probs + 1e-10), b_probs, reduction='sum').item()
        kl_divs.append(min(kl, 100.0))
        b_top10 = set(torch.topk(b_probs, 10).indices.tolist())
        m_top10 = set(torch.topk(m_probs, 10).indices.tolist())
        top_k_overlaps.append(len(b_top10 & m_top10) / 10.0)

    return torch.tensor([agreement_rate, np.mean(kl_divs), np.mean(top_k_overlaps)], dtype=torch.float32)


def compute_temporal_dynamics(logits_list: List[torch.Tensor]) -> torch.Tensor:
    """Entropy slope over time (2 dims)."""
    entropies = []
    for logits in logits_list:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)
    if len(entropies) < 2:
        return torch.tensor([0.0, 0.0], dtype=torch.float32)
    e_early = entropies[min(1, len(entropies) - 1)]
    e_mid = entropies[min(4, len(entropies) - 1)]
    e_late = entropies[min(9, len(entropies) - 1)]
    slope_early_mid = (e_mid - e_early) / 3.0
    slope_mid_late = (e_late - e_mid) / 5.0
    return torch.tensor([slope_early_mid, slope_mid_late], dtype=torch.float32)


# ============================================================
# Main Evaluator
# ============================================================

class FOKAdaptiveEvaluator:
    """
    V5 Evaluator that uses FOK probe for value-aware routing.
    Extracts all 45 features at inference time.
    """

    def __init__(
        self,
        model_path: str,
        probe_path: str,
        probe_config_path: str,
        calibration_dir: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
        utility_threshold: float = 0.0,
        layers: List[int] = [8, 16, 24, 32],
    ):
        self.utility_threshold = utility_threshold
        self.layers = layers
        self.feature_mean = None
        self.feature_std = None

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

        # Load probe config
        with open(probe_config_path, 'r') as f:
            config = json.load(f)

        # Load probe
        print(f"Loading probe: {probe_path}")
        if config.get('use_ensemble', False):
            self.probe = EnsembleFOKProbe(
                num_probes=config['num_probes'],
                input_dim=config['input_dim'],
            )
        else:
            self.probe = FOKProbe(input_dim=config['input_dim'])

        self.probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
        self.probe.eval()
        self.probe.to(self.device)

        # Load normalization stats (REQUIRED: features must be normalized the same way as training)
        if norm_stats_path and os.path.exists(norm_stats_path):
            print(f"Loading normalization stats: {norm_stats_path}")
            norm_stats = torch.load(norm_stats_path, map_location='cpu')
            self.feature_mean = norm_stats['feature_mean'].to(self.device)
            self.feature_std = norm_stats['feature_std'].to(self.device)
            print(f"  Feature mean range: [{self.feature_mean.min().item():.4f}, {self.feature_mean.max().item():.4f}]")
            print(f"  Feature std range:  [{self.feature_std.min().item():.4f}, {self.feature_std.max().item():.4f}]")
        else:
            print("WARNING: No norm_stats_path provided. Features will NOT be normalized.")
            print("         This will cause the probe to produce near-zero utility scores (zero-routing bug).")
            print("         Pass --norm_stats_path models/acc_v5_final/norm_stats.pt to fix this.")

        # Load calibration data (KNN)
        self.knn_index = None
        self.cal_embeddings = None
        self.cal_metadata = None

        if calibration_dir and os.path.exists(calibration_dir):
            self._load_calibration_data(calibration_dir)

    def _load_calibration_data(self, calibration_dir: str):
        """Load KNN calibration data from training."""
        print(f"Loading calibration data from: {calibration_dir}")
        from sklearn.neighbors import NearestNeighbors

        all_embeddings = []
        all_metadata = []

        for benchmark in ["gsm8k", "mmlu", "hellaswag"]:
            emb_path = os.path.join(calibration_dir, f"{benchmark}_embeddings.npy")
            meta_path = os.path.join(calibration_dir, f"{benchmark}_metadata.json")

            if os.path.exists(emb_path) and os.path.exists(meta_path):
                embeddings = np.load(emb_path)
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                all_embeddings.append(embeddings)
                all_metadata.extend(metadata)
                print(f"  Loaded {len(metadata)} calibration samples from {benchmark}")

        if all_embeddings:
            self.cal_embeddings = np.vstack(all_embeddings)
            self.cal_metadata = all_metadata
            self.knn_index = NearestNeighbors(
                n_neighbors=min(10, len(self.cal_embeddings)), metric='cosine'
            )
            self.knn_index.fit(self.cal_embeddings)
            print(f"  KNN index built with {len(self.cal_embeddings)} samples")
        else:
            print("  No calibration data found, using zeros for calibration features")

    def _encode_question(self, question: str) -> np.ndarray:
        """Encode a single question for KNN lookup."""
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_encoder'):
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            return self._encoder.encode([question])[0]
        except ImportError:
            from sklearn.feature_extraction.text import TfidfVectorizer
            if not hasattr(self, '_tfidf'):
                self._tfidf = TfidfVectorizer(max_features=384)
                if self.cal_metadata:
                    questions = [m['question'] for m in self.cal_metadata]
                    self._tfidf.fit(questions)
            return self._tfidf.transform([question]).toarray()[0]

    def _compute_calibration_features(self, question: str) -> torch.Tensor:
        """Compute KNN calibration features (3 dims)."""
        if self.knn_index is None:
            return torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32)

        query_emb = self._encode_question(question)
        k = min(10, len(self.cal_metadata))
        distances, indices = self.knn_index.kneighbors(query_emb.reshape(1, -1), n_neighbors=k)

        baseline_accs, metacog_accs, utilities = [], [], []
        for idx in indices[0]:
            m = self.cal_metadata[idx]
            baseline_accs.append(1.0 if m['baseline_correct'] else 0.0)
            metacog_accs.append(1.0 if m['metacog_correct'] else 0.0)
            utilities.append(m['utility_gain'])

        return torch.tensor([
            np.mean(baseline_accs),
            np.mean(metacog_accs),
            np.mean(utilities),
        ], dtype=torch.float32)

    def _extract_hidden_states(self, prompt: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Extract hidden states and answer-position logits."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = {}
        for layer_idx in self.layers:
            if layer_idx < len(outputs.hidden_states):
                h = outputs.hidden_states[layer_idx][0, -1, :].cpu()
                hidden_states[f"layer_{layer_idx}"] = h

        logits = outputs.logits[0, -1, :].cpu()
        return hidden_states, logits

    def _generate_tokens_with_logits(self, prompt: str, max_tokens: int = 10) -> Tuple[List[int], List[torch.Tensor]]:
        """Generate tokens and return token IDs + per-token logits."""
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
                max_new_tokens=max_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:].cpu().tolist()
        all_logits = [score[0].cpu() for score in outputs.scores[:max_tokens]]
        return generated_ids, all_logits

    def extract_all_features(self, question: str, benchmark: str) -> torch.Tensor:
        """
        Extract all 45 FOK features for a question.
        This runs both baseline and metacog feature extraction streams.
        """
        baseline_prompt = self._format_baseline_prompt(question, benchmark)
        metacog_prompt = self._format_metacog_prompt(question, benchmark)

        # ========================================
        # BASELINE STREAM
        # ========================================
        baseline_hidden, baseline_answer_logits = self._extract_hidden_states(baseline_prompt)
        baseline_dynamic = compute_dynamic_features(baseline_hidden)
        baseline_tokens, baseline_logits = self._generate_tokens_with_logits(baseline_prompt, max_tokens=10)
        baseline_early_entropy = compute_early_entropy(baseline_logits, num_tokens=2)
        baseline_gap = compute_answer_position_gap(baseline_answer_logits)
        baseline_layer_entropy = compute_layer_entropy_logit_lens(self.model, baseline_hidden, self.layers)
        baseline_temporal = compute_temporal_dynamics(baseline_logits)

        # ========================================
        # METACOG STREAM
        # ========================================
        metacog_hidden, metacog_answer_logits = self._extract_hidden_states(metacog_prompt)
        metacog_dynamic = compute_dynamic_features(metacog_hidden)
        metacog_tokens, metacog_logits = self._generate_tokens_with_logits(metacog_prompt, max_tokens=10)
        metacog_early_entropy = compute_early_entropy(metacog_logits, num_tokens=2)
        metacog_gap = compute_answer_position_gap(metacog_answer_logits)
        metacog_layer_entropy = compute_layer_entropy_logit_lens(self.model, metacog_hidden, self.layers)
        metacog_temporal = compute_temporal_dynamics(metacog_logits)

        # ========================================
        # CROSS-COMPARISON FEATURES
        # ========================================
        metacog_div_speed = compute_divergence_speed(metacog_logits)

        baseline_first = baseline_tokens[0] if baseline_tokens else 0
        metacog_first = metacog_tokens[0] if metacog_tokens else 0
        cross_comparison = compute_cross_comparison(
            baseline_first, metacog_first, baseline_gap, metacog_gap
        )

        answer_agreement = compute_early_answer_agreement(
            baseline_tokens, metacog_tokens,
            baseline_logits, metacog_logits,
            num_tokens=5
        )

        # ========================================
        # CALIBRATION FEATURES
        # ========================================
        calibration = self._compute_calibration_features(question)

        # ========================================
        # ASSEMBLE 45-DIM FEATURE VECTOR
        # ========================================
        features = torch.cat([
            baseline_dynamic,           # 9 dims  [0:9]
            baseline_early_entropy,     # 2 dims  [9:11]
            metacog_dynamic,            # 9 dims  [11:20]
            metacog_early_entropy,      # 2 dims  [20:22]
            torch.tensor([baseline_gap, metacog_gap], dtype=torch.float32),  # [22:24]
            torch.tensor([metacog_div_speed], dtype=torch.float32),          # [24]
            baseline_layer_entropy,     # 4 dims  [25:29]
            metacog_layer_entropy,      # 4 dims  [29:33]
            cross_comparison,           # 2 dims  [33:35]
            answer_agreement,           # 3 dims  [35:38]
            baseline_temporal,          # 2 dims  [38:40]
            metacog_temporal,           # 2 dims  [40:42]
            calibration,               # 3 dims  [42:45]
        ])

        return features

    def predict_utility(self, question: str, benchmark: str) -> Tuple[float, float, float]:
        """Predict utility gain using all 45 FOK features."""
        features = self.extract_all_features(question, benchmark)
        features = features.unsqueeze(0).to(self.device)

        # Apply the same normalization used during training (critical for correct routing)
        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / self.feature_std
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features = torch.clamp(features, -10.0, 10.0)

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
        return f"""{question}\n\nProvide a clear answer.\nFinal Answer: [answer]"""

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

    def evaluate_sample(self, question: str, ground_truth: str, benchmark: str) -> Dict:
        """Evaluate a single sample with FOK-based routing."""
        p_wrong, conflict, utility = self.predict_utility(question, benchmark)
        use_metacog = utility > self.utility_threshold

        if use_metacog:
            prompt = self._format_metacog_prompt(question, benchmark)
            max_tokens = 1024
        else:
            prompt = self._format_baseline_prompt(question, benchmark)
            max_tokens = 512

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

    def evaluate_benchmark(self, benchmark: str, num_samples: int = 1000) -> Dict:
        """Evaluate on a benchmark."""
        from datasets import load_dataset

        print(f"\nLoading {benchmark}...")
        samples = []

        if benchmark == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="test")
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            for idx in indices:
                item = dataset[idx]
                answer = item['answer'].split('####')[-1].strip()
                samples.append({'question': item['question'], 'answer': answer})

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

        results = []
        for sample in tqdm(samples, desc=f"Evaluating {benchmark}"):
            result = self.evaluate_sample(sample['question'], sample['answer'], benchmark)
            results.append(result)

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
    parser = argparse.ArgumentParser(description="FOK-Inspired Adaptive Evaluator (V5)")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--probe_config_path", type=str, required=True)
    parser.add_argument("--calibration_dir", type=str, default=None,
                        help="Directory containing training embeddings and metadata for KNN calibration")
    parser.add_argument("--norm_stats_path", type=str, default=None,
                        help="Path to norm_stats.pt from training (e.g. models/acc_v5_final/norm_stats.pt). "
                             "Required for correct routing — without this, probe outputs near-zero utility for all samples.")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "hellaswag"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--utility_threshold", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = FOKAdaptiveEvaluator(
        model_path=args.model_path,
        probe_path=args.probe_path,
        probe_config_path=args.probe_config_path,
        calibration_dir=args.calibration_dir,
        norm_stats_path=args.norm_stats_path,
        utility_threshold=args.utility_threshold,
    )

    all_summaries = []

    for benchmark in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"FOK-ADAPTIVE V5 EVALUATION: {benchmark.upper()}")
        print(f"Utility threshold: {args.utility_threshold}")
        print(f"{'='*60}")

        result = evaluator.evaluate_benchmark(benchmark, args.num_samples)

        s = result['summary']
        print(f"\n{'='*60}")
        print(f"RESULTS: {benchmark.upper()}")
        print(f"{'='*60}")
        print(f"Overall accuracy: {s['accuracy']*100:.2f}% ({s['correct']}/{s['total']})")
        print(f"\nMethod distribution:")
        print(f"  Baseline: {s['baseline_count']} ({s['baseline_count']/s['total']*100:.1f}%)")
        print(f"  Metacognition: {s['metacog_count']} ({s['metacog_count']/s['total']*100:.1f}%)")
        print(f"\nAccuracy by method:")
        if s['baseline_count'] > 0:
            print(f"  Baseline: {s['baseline_accuracy']*100:.2f}% ({s['baseline_correct']}/{s['baseline_count']})")
        if s['metacog_count'] > 0:
            print(f"  Metacognition: {s['metacog_accuracy']*100:.2f}% ({s['metacog_correct']}/{s['metacog_count']})")
        print(f"{'='*60}")

        all_summaries.append(s)

        result_path = os.path.join(args.output_dir, f"fok_v5_{benchmark}_results.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {result_path}")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for s in all_summaries:
        print(f"\n{s['benchmark'].upper()}:")
        print(f"  Accuracy: {s['accuracy']*100:.2f}%")
        print(f"  Baseline used: {s['baseline_count']} ({s['baseline_accuracy']*100:.1f}% acc)")
        print(f"  Metacog used: {s['metacog_count']} ({s['metacog_accuracy']*100:.1f}% acc)")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
