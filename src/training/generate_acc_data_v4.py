"""
ACC-Inspired Data Generation (V4)

Generates training data with:
1. Compressed hidden states (PCA: 16384 -> 256 dims)
2. Dynamic features (cosine drift, norm ratios, residual changes)
3. Early branching entropy (2 tokens, not 3-5 samples)
4. Utility gain labels (accuracy improvement - cost)

Key innovation: Value-aware labels, not just difficulty prediction.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA


def extract_hidden_states_multi_layer(
    model,
    tokenizer,
    question: str,
    layers: List[int] = [8, 16, 24, 32],
) -> Dict[str, torch.Tensor]:
    """Extract hidden states from multiple layers."""
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = {}
    for layer_idx in layers:
        if layer_idx < len(outputs.hidden_states):
            # Get last token position
            h = outputs.hidden_states[layer_idx][0, -1, :].cpu()
            hidden_states[f"layer_{layer_idx}"] = h
    
    return hidden_states


def compute_dynamic_features(hidden_states: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute dynamic features: how representations change across layers.
    
    Features:
    - Cosine drift: 1 - cos(h[L], h[L-1])
    - Norm ratio: ||h[L]|| / ||h[L-1]||
    - Residual change: ||h[L] - h[L-1]|| / ||h[L-1]||
    """
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


def compute_early_branching_entropy(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 2,
) -> Dict[str, float]:
    """
    Generate 1-2 tokens and measure early uncertainty.
    Much cheaper than full self-consistency (3-5 samples).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )
    
    # Compute entropy at each early token
    entropies = []
    for i, logits in enumerate(outputs.scores[:max_tokens]):
        probs = torch.softmax(logits[0], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)
    
    if len(entropies) == 0:
        return {'mean_entropy': 0.0, 'max_entropy': 0.0}
    
    return {
        'mean_entropy': np.mean(entropies),
        'max_entropy': np.max(entropies),
    }


def compute_utility_gain(
    baseline_correct: bool,
    metacog_correct: bool,
    compute_cost: float = 0.3,
) -> float:
    """
    Compute utility gain for value-aware training.
    
    Utility = Accuracy gain - Cost
    
    Args:
        baseline_correct: Whether baseline answer is correct
        metacog_correct: Whether metacog answer is correct
        compute_cost: Cost of using metacog (relative to baseline)
    
    Returns:
        Utility gain (positive = metacog worth it, negative = not worth it)
    """
    if baseline_correct and metacog_correct:
        accuracy_gain = 0.0  # Both correct, no improvement
    elif not baseline_correct and metacog_correct:
        accuracy_gain = 1.0  # Metacog fixes error
    elif baseline_correct and not metacog_correct:
        accuracy_gain = -1.0  # Metacog breaks correct answer
    else:
        accuracy_gain = 0.0  # Both wrong, no improvement
    
    utility_gain = accuracy_gain - compute_cost
    
    return utility_gain


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


def extract_answer(response: str, benchmark: str) -> str:
    """Extract answer from model response."""
    import re
    
    response = response.strip()
    
    # Look for "Final Answer: X" pattern
    final_match = re.search(r'Final Answer:\s*([^\n]+)', response, re.IGNORECASE)
    if final_match:
        answer = final_match.group(1).strip()
        # Clean up
        answer = re.sub(r'[^\w\d\.\-]', '', answer)
        return answer
    
    if benchmark == "gsm8k":
        # Look for numbers
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]
    
    elif benchmark in ["mmlu", "hellaswag"]:
        # Look for letters
        letters = re.findall(r'\b([A-D])\b', response.upper())
        if letters:
            return letters[-1]
    
    return ""


def check_answer(predicted: str, ground_truth: str, benchmark: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred = predicted.strip().upper()
    gt = ground_truth.strip().upper()
    
    if benchmark == "gsm8k":
        # Numeric comparison
        try:
            pred_num = float(re.sub(r'[^\d\.\-]', '', pred))
            gt_num = float(re.sub(r'[^\d\.\-]', '', gt))
            return abs(pred_num - gt_num) < 0.01
        except:
            return pred == gt
    
    else:
        # Letter comparison
        return pred == gt


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate answer from model."""
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


def load_benchmark_samples(benchmark: str, num_samples: int, split: str = "train") -> List[Dict]:
    """Load samples from benchmark with stratified sampling for MMLU."""
    from datasets import load_dataset
    
    samples = []
    
    if benchmark == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split=split)
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        for idx in indices:
            item = dataset[idx]
            # Extract answer from solution
            answer = item['answer'].split('####')[-1].strip()
            samples.append({
                'question': item['question'],
                'answer': answer,
                'benchmark': 'gsm8k',
            })
    
    elif benchmark == "mmlu":
        # Stratified sampling across subjects
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
                    # Format question with options
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
            # Format question with options
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


def main():
    parser = argparse.ArgumentParser(description="Generate ACC-inspired training data (V4)")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "hellaswag"])
    parser.add_argument("--samples_per_benchmark", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="data/training/acc_v4")
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24, 32])
    parser.add_argument("--compressed_dim", type=int, default=256)
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
        print(f"Processing {benchmark.upper()}")
        print(f"{'='*60}")
        
        # Load samples
        print(f"Loading {args.samples_per_benchmark} samples...")
        samples = load_benchmark_samples(benchmark, args.samples_per_benchmark)
        print(f"Loaded {len(samples)} samples")
        
        # Collect data
        all_hidden_states = []
        all_dynamic_features = []
        all_early_entropy = []
        all_wrong_labels = []
        all_conflict_labels = []
        all_utility_labels = []
        metadata = []
        
        for sample in tqdm(samples, desc=f"Generating {benchmark} data"):
            question = sample['question']
            ground_truth = sample['answer']
            
            try:
                # 1. Extract multi-layer hidden states
                hidden_states = extract_hidden_states_multi_layer(
                    model, tokenizer, question, layers=args.layers
                )
                
                # Concatenate all layers
                concat_hidden = torch.cat([hidden_states[f"layer_{l}"] for l in args.layers])
                all_hidden_states.append(concat_hidden)
                
                # 2. Compute dynamic features
                dynamic_feats = compute_dynamic_features(hidden_states)
                all_dynamic_features.append(dynamic_feats)
                
                # 3. Compute early branching entropy
                baseline_prompt = format_baseline_prompt(question, benchmark)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": baseline_prompt}
                ]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                entropy_feats = compute_early_branching_entropy(model, tokenizer, formatted_prompt)
                all_early_entropy.append(torch.tensor([
                    entropy_feats['mean_entropy'],
                    entropy_feats['max_entropy']
                ], dtype=torch.float32))
                
                # 4. Generate baseline answer
                baseline_response = generate_answer(model, tokenizer, baseline_prompt)
                baseline_answer = extract_answer(baseline_response, benchmark)
                baseline_correct = check_answer(baseline_answer, ground_truth, benchmark)
                
                # 5. Generate metacog answer
                metacog_prompt = format_metacog_prompt(question, benchmark)
                metacog_response = generate_answer(model, tokenizer, metacog_prompt, max_new_tokens=1024)
                metacog_answer = extract_answer(metacog_response, benchmark)
                metacog_correct = check_answer(metacog_answer, ground_truth, benchmark)
                
                # 6. Compute labels
                wrong_label = 0.0 if baseline_correct else 1.0
                conflict_label = entropy_feats['mean_entropy'] / 10.0  # Normalize
                utility_label = compute_utility_gain(baseline_correct, metacog_correct)
                
                all_wrong_labels.append(wrong_label)
                all_conflict_labels.append(conflict_label)
                all_utility_labels.append(utility_label)
                
                # Save metadata
                metadata.append({
                    'question': question[:200],
                    'ground_truth': ground_truth,
                    'baseline_answer': baseline_answer,
                    'metacog_answer': metacog_answer,
                    'baseline_correct': baseline_correct,
                    'metacog_correct': metacog_correct,
                    'utility_gain': utility_label,
                })
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        # Stack tensors
        hidden_states_tensor = torch.stack(all_hidden_states)
        dynamic_features_tensor = torch.stack(all_dynamic_features)
        early_entropy_tensor = torch.stack(all_early_entropy)
        
        print(f"\nHidden states shape: {hidden_states_tensor.shape}")
        print(f"Dynamic features shape: {dynamic_features_tensor.shape}")
        print(f"Early entropy shape: {early_entropy_tensor.shape}")
        
        # Apply PCA compression
        print(f"\nApplying PCA compression: {hidden_states_tensor.shape[1]} -> {args.compressed_dim}")
        pca = PCA(n_components=args.compressed_dim)
        compressed_hidden = pca.fit_transform(hidden_states_tensor.numpy())
        compressed_hidden_tensor = torch.tensor(compressed_hidden, dtype=torch.float32)
        print(f"Compressed hidden states shape: {compressed_hidden_tensor.shape}")
        print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.3f}")
        
        # Save PCA model
        import pickle
        pca_path = os.path.join(args.output_dir, f"{benchmark}_pca.pkl")
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"Saved PCA model to: {pca_path}")
        
        # Combine all features
        all_features = torch.cat([
            compressed_hidden_tensor,  # 256 dims
            dynamic_features_tensor,    # 9 dims (3 layers × 3 features)
            early_entropy_tensor,       # 2 dims
        ], dim=1)
        
        print(f"Total feature dimensions: {all_features.shape[1]}")
        
        # Save tensors
        tensor_data = {
            'features': all_features,
            'wrong_labels': torch.tensor(all_wrong_labels, dtype=torch.float32),
            'conflict_labels': torch.tensor(all_conflict_labels, dtype=torch.float32),
            'utility_labels': torch.tensor(all_utility_labels, dtype=torch.float32),
            'raw_hidden_states': hidden_states_tensor,  # Keep for ablation
        }
        
        tensor_path = os.path.join(args.output_dir, f"{benchmark}_tensors.pt")
        torch.save(tensor_data, tensor_path)
        print(f"Saved tensors to: {tensor_path}")
        
        # Save metadata
        metadata_path = os.path.join(args.output_dir, f"{benchmark}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_path}")
        
        # Print statistics
        wrong_rate = np.mean(all_wrong_labels)
        utility_positive = sum(1 for u in all_utility_labels if u > 0) / len(all_utility_labels)
        print(f"\nStatistics for {benchmark}:")
        print(f"  Baseline error rate: {wrong_rate*100:.1f}%")
        print(f"  Utility positive rate: {utility_positive*100:.1f}%")
        print(f"  Mean utility: {np.mean(all_utility_labels):.3f}")
    
    print(f"\n{'='*60}")
    print("Data generation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import re
    main()
