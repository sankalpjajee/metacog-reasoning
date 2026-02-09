"""
ACC-Inspired Data Generation (V5) - Feeling of Knowing (FOK)

Generates training data with 45-dimensional feature set:

Part 1: Baseline Features (11 dims)
  - Dynamic features: cosine drift, norm ratio, residual change x 3 transitions (9)
  - Early entropy: mean, max of first 2 reasoning tokens (2)

Part 2: Metacog Features (11 dims)
  - Same structure as baseline but from metacog prompt

Part 3: Answer-Position Features (2 dims)
  - Top-1 vs Top-2 probability gap at "Final Answer:" position
  - For both baseline and metacog prompts

Part 4: Metacog Divergence Speed (1 dim)
  - How quickly metacog resolves uncertainty (entropy token1 - token2)

Part 5: Layer Entropy via Logit Lens (8 dims)
  - Project hidden states at layers 8,16,24,32 through LM head
  - Compute entropy of resulting distribution
  - For both baseline and metacog

Part 6: Cross-Comparison (2 dims)
  - Token divergence: do baseline/metacog produce different first tokens?
  - Answer gap difference: metacog_gap - baseline_gap

Part 7: Early Answer Agreement (3 dims)
  - Token agreement rate (5 tokens)
  - Mean KL divergence between logit distributions
  - Mean top-10 overlap

Part 8: Temporal Dynamics (4 dims)
  - Entropy slope early->mid (tokens 2->5) for baseline and metacog
  - Entropy slope mid->late (tokens 5->10) for baseline and metacog

Part 9: Confidence Calibration (3 dims)
  - Historical baseline accuracy on similar questions (KNN)
  - Historical metacog accuracy on similar questions (KNN)
  - Historical utility on similar questions (KNN)

Total: 11 + 11 + 2 + 1 + 8 + 2 + 3 + 4 + 3 = 45 dims

Key innovation: Replicates the brain's Feeling of Knowing (FOK) mechanism
by comparing baseline and metacognitive processing streams.
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
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
# Part 1 & 2: Dynamic Features + Early Entropy (Baseline & Metacog)
# ============================================================

def extract_hidden_states_multi_layer(
    model,
    tokenizer,
    prompt: str,
    layers: List[int] = [8, 16, 24, 32],
    is_chat: bool = True,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Extract hidden states from multiple layers for a given prompt.
    
    Returns:
        hidden_states: dict mapping layer names to hidden state tensors
        logits: logits at the last token position (for answer-position gap)
    """
    if is_chat:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = {}
    for layer_idx in layers:
        if layer_idx < len(outputs.hidden_states):
            h = outputs.hidden_states[layer_idx][0, -1, :].cpu()
            hidden_states[f"layer_{layer_idx}"] = h

    logits = outputs.logits[0, -1, :].cpu()

    return hidden_states, logits


def compute_dynamic_features(hidden_states: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute dynamic features: how representations change across layers.

    Features per transition (3 transitions x 3 features = 9 dims):
    - Cosine drift: 1 - cos(h[L], h[L-1])
    - Norm ratio: ||h[L]|| / ||h[L-1]||
    - Residual change: ||h[L] - h[L-1]|| / ||h[L-1]||
    """
    layers = sorted([int(k.split("_")[1]) for k in hidden_states.keys()])
    features = []

    for i in range(1, len(layers)):
        h_prev = hidden_states[f"layer_{layers[i-1]}"]
        h_curr = hidden_states[f"layer_{layers[i]}"]

        cosine_sim = F.cosine_similarity(h_prev.unsqueeze(0), h_curr.unsqueeze(0))
        cosine_drift = 1.0 - cosine_sim.item()
        features.append(cosine_drift)

        norm_prev = torch.norm(h_prev).item()
        norm_curr = torch.norm(h_curr).item()
        norm_ratio = norm_curr / (norm_prev + 1e-8)
        features.append(norm_ratio)

        residual = h_curr - h_prev
        residual_norm = torch.norm(residual).item()
        residual_change = residual_norm / (norm_prev + 1e-8)
        features.append(residual_change)

    return torch.tensor(features, dtype=torch.float32)


def generate_tokens_with_logits(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 10,
) -> Tuple[List[int], List[torch.Tensor]]:
    """
    Generate tokens and return both token IDs and per-token logits.
    Used for early entropy, temporal dynamics, and answer agreement.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )

    generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:].cpu().tolist()

    all_logits = []
    for score in outputs.scores[:max_tokens]:
        all_logits.append(score[0].cpu())

    return generated_ids, all_logits


def compute_early_entropy(logits_list: List[torch.Tensor], num_tokens: int = 2) -> torch.Tensor:
    """Compute early entropy from first N token logits. Returns [mean, max]."""
    entropies = []
    for logits in logits_list[:num_tokens]:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)

    if len(entropies) == 0:
        return torch.tensor([0.0, 0.0], dtype=torch.float32)

    return torch.tensor([np.mean(entropies), np.max(entropies)], dtype=torch.float32)


# ============================================================
# Part 3: Answer-Position Gap
# ============================================================

def compute_answer_position_gap(logits: torch.Tensor) -> float:
    """
    Compute Top-1 vs Top-2 probability gap at the answer position.
    
    This is measured at the LAST token position of the prompt
    (right before the model would generate "Final Answer: X").
    
    Small gap = competing candidates = FOK signal
    Large gap = confident (right or wrong)
    """
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, _ = torch.sort(probs, descending=True)
    gap = (sorted_probs[0] - sorted_probs[1]).item()
    return gap


# ============================================================
# Part 4: Metacog Divergence Speed
# ============================================================

def compute_divergence_speed(logits_list: List[torch.Tensor]) -> float:
    """
    How quickly metacog resolves uncertainty.
    = entropy(token1) - entropy(token2)
    
    Large positive = fast resolution = FOK
    """
    if len(logits_list) < 2:
        return 0.0

    entropies = []
    for logits in logits_list[:2]:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)

    return entropies[0] - entropies[1]


# ============================================================
# Part 5: Layer Entropy via Logit Lens
# ============================================================

def compute_layer_entropy_logit_lens(
    model,
    hidden_states: Dict[str, torch.Tensor],
    layers: List[int] = [8, 16, 24, 32],
) -> torch.Tensor:
    """
    Project each layer's hidden state through the LM head to get
    entropy at each layer. This reveals how certainty evolves
    through the network.
    
    Pattern: up-then-down = FOK (explored then found it)
    Monotonic decrease = confident from start
    Monotonic increase = lost and staying lost
    """
    lm_head = model.lm_head
    entropies = []

    for layer_idx in layers:
        key = f"layer_{layer_idx}"
        if key not in hidden_states:
            entropies.append(0.0)
            continue

        h = hidden_states[key].to(lm_head.weight.device)

        # Apply layer norm if model has one
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            h = model.model.norm(h.unsqueeze(0)).squeeze(0)

        # Convert to same dtype as lm_head weights
        h = h.to(lm_head.weight.dtype)

        with torch.no_grad():
            logits = lm_head(h)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            entropies.append(entropy)

    return torch.tensor(entropies, dtype=torch.float32)


# ============================================================
# Part 6: Cross-Comparison Features
# ============================================================

def compute_cross_comparison(
    baseline_first_token: int,
    metacog_first_token: int,
    baseline_gap: float,
    metacog_gap: float,
) -> torch.Tensor:
    """
    Cross-comparison between baseline and metacog:
    - Token divergence: do they produce different first reasoning tokens? (binary)
    - Answer gap difference: metacog_gap - baseline_gap (positive = metacog resolved)
    """
    token_divergence = 1.0 if baseline_first_token != metacog_first_token else 0.0
    gap_difference = metacog_gap - baseline_gap

    return torch.tensor([token_divergence, gap_difference], dtype=torch.float32)


# ============================================================
# Part 7: Early Answer Agreement (5 tokens)
# ============================================================

def compute_early_answer_agreement(
    baseline_tokens: List[int],
    metacog_tokens: List[int],
    baseline_logits: List[torch.Tensor],
    metacog_logits: List[torch.Tensor],
    num_tokens: int = 5,
) -> torch.Tensor:
    """
    Compare first 5 generated tokens between baseline and metacog:
    - Token agreement rate: fraction of matching tokens
    - Mean KL divergence: average KL(baseline || metacog)
    - Mean top-10 overlap: average overlap in top-10 candidates
    """
    n = min(num_tokens, len(baseline_tokens), len(metacog_tokens),
            len(baseline_logits), len(metacog_logits))

    if n == 0:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    matches = sum(1 for i in range(n) if baseline_tokens[i] == metacog_tokens[i])
    agreement_rate = matches / n

    kl_divs = []
    top_k_overlaps = []

    for i in range(n):
        b_probs = torch.softmax(baseline_logits[i], dim=-1)
        m_probs = torch.softmax(metacog_logits[i], dim=-1)

        kl = F.kl_div(
            torch.log(m_probs + 1e-10),
            b_probs,
            reduction='sum'
        ).item()
        kl_divs.append(min(kl, 100.0))  # Clip extreme values

        b_top10 = set(torch.topk(b_probs, 10).indices.tolist())
        m_top10 = set(torch.topk(m_probs, 10).indices.tolist())
        overlap = len(b_top10 & m_top10) / 10.0
        top_k_overlaps.append(overlap)

    return torch.tensor([
        agreement_rate,
        np.mean(kl_divs),
        np.mean(top_k_overlaps),
    ], dtype=torch.float32)


# ============================================================
# Part 8: Temporal Dynamics
# ============================================================

def compute_temporal_dynamics(logits_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Measure how entropy changes over time during generation.
    
    Generates 10 tokens, measures entropy at tokens 2, 5, 10.
    Returns 2 slopes: early->mid, mid->late.
    
    Increasing entropy = getting more uncertain (lost)
    Decreasing entropy = converging (finding answer)
    """
    entropies = []
    for logits in logits_list:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)

    if len(entropies) < 2:
        return torch.tensor([0.0, 0.0], dtype=torch.float32)

    e_early = entropies[min(1, len(entropies) - 1)]   # Token 2
    e_mid = entropies[min(4, len(entropies) - 1)]      # Token 5
    e_late = entropies[min(9, len(entropies) - 1)]     # Token 10

    slope_early_mid = (e_mid - e_early) / 3.0
    slope_mid_late = (e_late - e_mid) / 5.0

    return torch.tensor([slope_early_mid, slope_mid_late], dtype=torch.float32)


# ============================================================
# Part 9: Confidence Calibration from Experience (KNN)
# ============================================================

def build_calibration_index(metadata: List[Dict], embeddings: np.ndarray):
    """Build a KNN index for confidence calibration."""
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=min(10, len(embeddings)), metric='cosine')
    knn.fit(embeddings)
    return knn


def compute_calibration_features(
    query_embedding: np.ndarray,
    knn_index,
    all_embeddings: np.ndarray,
    metadata: List[Dict],
    k: int = 10,
) -> torch.Tensor:
    """
    Find K nearest neighbors and compute historical performance.
    
    Returns:
    - Historical baseline accuracy on similar questions
    - Historical metacog accuracy on similar questions
    - Historical utility on similar questions
    """
    k = min(k, len(metadata))
    distances, indices = knn_index.kneighbors(query_embedding.reshape(1, -1), n_neighbors=k)

    baseline_accs = []
    metacog_accs = []
    utilities = []

    for idx in indices[0]:
        m = metadata[idx]
        baseline_accs.append(1.0 if m['baseline_correct'] else 0.0)
        metacog_accs.append(1.0 if m['metacog_correct'] else 0.0)
        utilities.append(m['utility_gain'])

    return torch.tensor([
        np.mean(baseline_accs),
        np.mean(metacog_accs),
        np.mean(utilities),
    ], dtype=torch.float32)


def encode_questions(questions: List[str]) -> np.ndarray:
    """
    Encode questions using a lightweight sentence transformer.
    Falls back to TF-IDF if sentence-transformers not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = encoder.encode(questions, show_progress_bar=True, batch_size=32)
        return embeddings
    except ImportError:
        print("sentence-transformers not available, using TF-IDF fallback")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=384)
        embeddings = vectorizer.fit_transform(questions).toarray()
        return embeddings


# ============================================================
# Prompt Formatting
# ============================================================

def format_baseline_prompt(question: str, benchmark: str) -> str:
    """Format baseline prompt for a question."""
    if benchmark == "gsm8k":
        return f"""Solve this math problem step by step.

Question: {question}

Provide your solution and end with "Final Answer: [number]"."""

    elif benchmark == "mmlu":
        return f"""{question}

Answer with the letter only (A, B, C, or D).
Final Answer: [letter]"""

    elif benchmark == "hellaswag":
        return f"""{question}

Answer with the letter only (A, B, C, or D).
Final Answer: [letter]"""

    return question


def format_metacog_prompt(question: str, benchmark: str) -> str:
    """Format metacognitive prompt for a question."""
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

    elif benchmark in ["mmlu", "hellaswag"]:
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

    return question


# ============================================================
# Answer Extraction & Checking
# ============================================================

def extract_answer(response: str, benchmark: str) -> str:
    """Extract answer from model response."""
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


def check_answer(predicted: str, ground_truth: str, benchmark: str) -> bool:
    """Check if predicted answer matches ground truth."""
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


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate full answer from model."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def compute_utility_gain(
    baseline_correct: bool,
    metacog_correct: bool,
    compute_cost: float = 0.3,
) -> float:
    """Compute utility gain for value-aware training."""
    if baseline_correct and metacog_correct:
        accuracy_gain = 0.0
    elif not baseline_correct and metacog_correct:
        accuracy_gain = 1.0
    elif baseline_correct and not metacog_correct:
        accuracy_gain = -1.0
    else:
        accuracy_gain = 0.0

    return accuracy_gain - compute_cost


# ============================================================
# Benchmark Loading
# ============================================================

def load_benchmark_samples(benchmark: str, num_samples: int, split: str = "train") -> List[Dict]:
    """Load samples from benchmark."""
    from datasets import load_dataset

    samples = []

    if benchmark == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split=split)
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        for idx in indices:
            item = dataset[idx]
            answer = item['answer'].split('####')[-1].strip()
            samples.append({
                'question': item['question'],
                'answer': answer,
                'benchmark': 'gsm8k',
            })

    elif benchmark == "mmlu":
        subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology", "high_school_statistics",
            "high_school_us_history", "high_school_world_history", "human_aging",
            "human_sexuality", "international_law", "jurisprudence",
            "logical_fallacies", "machine_learning", "management", "marketing",
            "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
            "nutrition", "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology", "us_foreign_policy",
            "virology", "world_religions"
        ]

        samples_per_subject = max(1, num_samples // len(subjects))

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
                    samples.append({
                        'question': formatted,
                        'answer': answer,
                        'benchmark': 'mmlu',
                        'subject': subject,
                    })
            except Exception as e:
                print(f"Warning: Could not load MMLU subject {subject}: {e}")
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
            samples.append({
                'question': formatted,
                'answer': answer,
                'benchmark': 'hellaswag',
            })

    return samples[:num_samples]


# ============================================================
# Main: Orchestrate All Feature Extraction
# ============================================================

def process_single_sample(
    model,
    tokenizer,
    sample: Dict,
    layers: List[int] = [8, 16, 24, 32],
) -> Optional[Dict]:
    """
    Process a single sample and extract all 42 features (before calibration).
    Calibration features (3 dims) are added after all samples are processed.
    """
    question = sample['question']
    ground_truth = sample['answer']
    benchmark = sample['benchmark']

    try:
        baseline_prompt = format_baseline_prompt(question, benchmark)
        metacog_prompt = format_metacog_prompt(question, benchmark)

        # ========================================
        # BASELINE STREAM
        # ========================================

        # Hidden states + answer-position logits
        baseline_hidden, baseline_answer_logits = extract_hidden_states_multi_layer(
            model, tokenizer, baseline_prompt, layers, is_chat=True
        )

        # Dynamic features (9 dims)
        baseline_dynamic = compute_dynamic_features(baseline_hidden)

        # Generate 10 tokens for entropy, temporal dynamics, agreement
        baseline_tokens, baseline_logits = generate_tokens_with_logits(
            model, tokenizer, baseline_prompt, max_tokens=10
        )

        # Early entropy (2 dims)
        baseline_early_entropy = compute_early_entropy(baseline_logits, num_tokens=2)

        # Answer-position gap (1 dim)
        baseline_gap = compute_answer_position_gap(baseline_answer_logits)

        # Layer entropy via logit lens (4 dims)
        baseline_layer_entropy = compute_layer_entropy_logit_lens(
            model, baseline_hidden, layers
        )

        # Temporal dynamics (2 dims)
        baseline_temporal = compute_temporal_dynamics(baseline_logits)

        # ========================================
        # METACOG STREAM
        # ========================================

        # Hidden states + answer-position logits
        metacog_hidden, metacog_answer_logits = extract_hidden_states_multi_layer(
            model, tokenizer, metacog_prompt, layers, is_chat=True
        )

        # Dynamic features (9 dims)
        metacog_dynamic = compute_dynamic_features(metacog_hidden)

        # Generate 10 tokens
        metacog_tokens, metacog_logits = generate_tokens_with_logits(
            model, tokenizer, metacog_prompt, max_tokens=10
        )

        # Early entropy (2 dims)
        metacog_early_entropy = compute_early_entropy(metacog_logits, num_tokens=2)

        # Answer-position gap (1 dim)
        metacog_gap = compute_answer_position_gap(metacog_answer_logits)

        # Layer entropy via logit lens (4 dims)
        metacog_layer_entropy = compute_layer_entropy_logit_lens(
            model, metacog_hidden, layers
        )

        # Temporal dynamics (2 dims)
        metacog_temporal = compute_temporal_dynamics(metacog_logits)

        # ========================================
        # CROSS-COMPARISON FEATURES
        # ========================================

        # Metacog divergence speed (1 dim)
        metacog_div_speed = compute_divergence_speed(metacog_logits)

        # Token divergence + gap difference (2 dims)
        baseline_first = baseline_tokens[0] if baseline_tokens else 0
        metacog_first = metacog_tokens[0] if metacog_tokens else 0
        cross_comparison = compute_cross_comparison(
            baseline_first, metacog_first, baseline_gap, metacog_gap
        )

        # Early answer agreement (3 dims)
        answer_agreement = compute_early_answer_agreement(
            baseline_tokens, metacog_tokens,
            baseline_logits, metacog_logits,
            num_tokens=5
        )

        # ========================================
        # GENERATE FULL ANSWERS FOR LABELS
        # ========================================

        baseline_response = generate_answer(model, tokenizer, baseline_prompt)
        baseline_answer = extract_answer(baseline_response, benchmark)
        baseline_correct = check_answer(baseline_answer, ground_truth, benchmark)

        metacog_response = generate_answer(model, tokenizer, metacog_prompt, max_new_tokens=1024)
        metacog_answer = extract_answer(metacog_response, benchmark)
        metacog_correct = check_answer(metacog_answer, ground_truth, benchmark)

        # ========================================
        # COMPUTE LABELS
        # ========================================

        utility_label = compute_utility_gain(baseline_correct, metacog_correct)
        wrong_label = 0.0 if baseline_correct else 1.0
        conflict_label = baseline_early_entropy[0].item() / 10.0

        # ========================================
        # ASSEMBLE FEATURE VECTOR (42 dims, before calibration)
        # ========================================

        features = torch.cat([
            # Part 1: Baseline features (11 dims)
            baseline_dynamic,           # 9 dims  [0:9]
            baseline_early_entropy,     # 2 dims  [9:11]

            # Part 2: Metacog features (11 dims)
            metacog_dynamic,            # 9 dims  [11:20]
            metacog_early_entropy,      # 2 dims  [20:22]

            # Part 3: Answer-position gap (2 dims)
            torch.tensor([baseline_gap, metacog_gap], dtype=torch.float32),  # [22:24]

            # Part 4: Metacog divergence speed (1 dim)
            torch.tensor([metacog_div_speed], dtype=torch.float32),  # [24]

            # Part 5: Layer entropy via logit lens (8 dims)
            baseline_layer_entropy,     # 4 dims  [25:29]
            metacog_layer_entropy,      # 4 dims  [29:33]

            # Part 6: Cross-comparison (2 dims)
            cross_comparison,           # 2 dims  [33:35]

            # Part 7: Early answer agreement (3 dims)
            answer_agreement,           # 3 dims  [35:38]

            # Part 8: Temporal dynamics (4 dims)
            baseline_temporal,          # 2 dims  [38:40]
            metacog_temporal,           # 2 dims  [40:42]
        ])

        metadata = {
            'question': question[:200],
            'ground_truth': ground_truth,
            'baseline_answer': baseline_answer,
            'metacog_answer': metacog_answer,
            'baseline_correct': baseline_correct,
            'metacog_correct': metacog_correct,
            'utility_gain': utility_label,
            'benchmark': benchmark,
        }

        return {
            'features': features,  # 42 dims (calibration added later)
            'wrong_label': wrong_label,
            'conflict_label': conflict_label,
            'utility_label': utility_label,
            'metadata': metadata,
        }

    except Exception as e:
        print(f"Error processing sample: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate ACC-inspired training data (V5) with FOK features")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "hellaswag"])
    parser.add_argument("--samples_per_benchmark", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="data/training/acc_v5")
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24, 32])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    # Process each benchmark
    for benchmark in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"Processing {benchmark.upper()} (V5 FOK)")
        print(f"{'='*60}")

        print(f"Loading {args.samples_per_benchmark} samples...")
        samples = load_benchmark_samples(benchmark, args.samples_per_benchmark)
        print(f"Loaded {len(samples)} samples")

        all_features = []
        all_wrong_labels = []
        all_conflict_labels = []
        all_utility_labels = []
        all_metadata = []

        for sample in tqdm(samples, desc=f"Generating {benchmark} V5 data"):
            result = process_single_sample(model, tokenizer, sample, layers=args.layers)

            if result is not None:
                all_features.append(result['features'])
                all_wrong_labels.append(result['wrong_label'])
                all_conflict_labels.append(result['conflict_label'])
                all_utility_labels.append(result['utility_label'])
                all_metadata.append(result['metadata'])

        if len(all_features) == 0:
            print(f"No valid samples for {benchmark}, skipping")
            continue

        features_tensor = torch.stack(all_features)
        print(f"\nFeatures shape (before calibration): {features_tensor.shape}")

        # ========================================
        # Part 9: Confidence Calibration (3 dims)
        # ========================================
        print("Computing confidence calibration features (KNN)...")

        questions = [m['question'] for m in all_metadata]
        embeddings = encode_questions(questions)

        knn_index = build_calibration_index(all_metadata, embeddings)

        calibration_features = []
        for i in range(len(all_metadata)):
            cal_feats = compute_calibration_features(
                embeddings[i], knn_index, embeddings, all_metadata, k=10
            )
            calibration_features.append(cal_feats)

        calibration_tensor = torch.stack(calibration_features)
        print(f"Calibration features shape: {calibration_tensor.shape}")

        # Combine all features (42 + 3 = 45 dims)
        full_features = torch.cat([features_tensor, calibration_tensor], dim=1)
        print(f"Full features shape: {full_features.shape}")

        # Save tensors
        tensor_data = {
            'features': full_features,  # 45 dims
            'wrong_labels': torch.tensor(all_wrong_labels, dtype=torch.float32),
            'conflict_labels': torch.tensor(all_conflict_labels, dtype=torch.float32),
            'utility_labels': torch.tensor(all_utility_labels, dtype=torch.float32),
        }

        tensor_path = os.path.join(args.output_dir, f"{benchmark}_tensors.pt")
        torch.save(tensor_data, tensor_path)
        print(f"Saved tensors to: {tensor_path}")

        # Save metadata
        metadata_path = os.path.join(args.output_dir, f"{benchmark}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_path}")

        # Save embeddings for evaluation-time calibration
        embeddings_path = os.path.join(args.output_dir, f"{benchmark}_embeddings.npy")
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to: {embeddings_path}")

        # Print statistics
        wrong_rate = np.mean(all_wrong_labels)
        utility_positive = sum(1 for u in all_utility_labels if u > 0) / len(all_utility_labels)
        print(f"\nStatistics for {benchmark}:")
        print(f"  Samples processed: {len(all_features)}")
        print(f"  Feature dimensions: {full_features.shape[1]}")
        print(f"  Baseline error rate: {wrong_rate*100:.1f}%")
        print(f"  Utility positive rate: {utility_positive*100:.1f}%")
        print(f"  Mean utility: {np.mean(all_utility_labels):.3f}")

        # Print feature group statistics
        print(f"\n  Feature group statistics:")
        print(f"    Baseline dynamic (0-8):     mean={full_features[:, 0:9].mean():.4f}")
        print(f"    Baseline entropy (9-10):     mean={full_features[:, 9:11].mean():.4f}")
        print(f"    Metacog dynamic (11-19):     mean={full_features[:, 11:20].mean():.4f}")
        print(f"    Metacog entropy (20-21):     mean={full_features[:, 20:22].mean():.4f}")
        print(f"    Answer gap (22-23):          mean={full_features[:, 22:24].mean():.4f}")
        print(f"    Divergence speed (24):       mean={full_features[:, 24].mean():.4f}")
        print(f"    Layer entropy base (25-28):  mean={full_features[:, 25:29].mean():.4f}")
        print(f"    Layer entropy meta (29-32):  mean={full_features[:, 29:33].mean():.4f}")
        print(f"    Cross-comparison (33-34):    mean={full_features[:, 33:35].mean():.4f}")
        print(f"    Answer agreement (35-37):    mean={full_features[:, 35:38].mean():.4f}")
        print(f"    Temporal baseline (38-39):   mean={full_features[:, 38:40].mean():.4f}")
        print(f"    Temporal metacog (40-41):    mean={full_features[:, 40:42].mean():.4f}")
        print(f"    Calibration (42-44):         mean={full_features[:, 42:45].mean():.4f}")

    print(f"\n{'='*60}")
    print("V5 FOK Data generation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Total feature dimensions: 45")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
