#!/usr/bin/env python3
"""
Generate metacognitive traces on MR-Ben dataset (harder than GSM8K).
Falls back to harder GSM8K problems if MR-Ben is not available.

Key difference from GSM8K version:
- Uses temperature 1.8 (instead of 1.2) for error-prone traces
- Targets harder multi-hop reasoning problems
"""

import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Metacognitive prompt template - encourages monitoring and control
PROMPT_METACOGNITIVE = """You are a careful problem solver. Solve this problem accurately.

As you work:
- Monitor your confidence: Note when you're certain or uncertain about steps
- Check your reasoning: Catch potential errors before they lead to wrong answers
- Verify your answer: Double-check calculations and logic

Problem: {problem}

Your response:"""

PROMPT_REASONING = """You are a helpful problem solver. Solve this problem step-by-step, showing your reasoning clearly.

Problem: {problem}

Your response:"""

PROMPT_DIRECT = """Solve this problem.

Problem: {problem}

Your response:"""


def load_model_and_tokenizer(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    return model, tokenizer


def load_dataset_with_fallback():
    """Try to load MR-Ben, fall back to harder GSM8K problems."""
    
    # Try MR-Ben first
    try:
        print("Attempting to load MR-Ben dataset...")
        dataset = load_dataset("MR-Ben/MR-Ben", split="train")
        print(f"✓ Loaded MR-Ben: {len(dataset)} examples")
        return dataset, "mrben"
    except Exception as e:
        print(f"Could not load MR-Ben: {e}")
    
    # Try alternative MR-Ben names
    for name in ["mrben", "mr-ben", "MRBen", "MR-Ben"]:
        try:
            print(f"Trying {name}...")
            dataset = load_dataset(name, split="train")
            print(f"✓ Loaded {name}: {len(dataset)} examples")
            return dataset, "mrben"
        except:
            continue
    
    # Fall back to harder GSM8K problems
    print("Falling back to GSM8K (filtering for harder problems)...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # Filter for longer/more complex problems
    harder_problems = []
    for example in dataset:
        question = example['question']
        # Heuristics for harder problems:
        # - Longer questions (more steps)
        # - More numbers (more calculations)
        # - More sentences (more complex)
        word_count = len(question.split())
        sentence_count = question.count('.') + question.count('?')
        
        if word_count > 40 or sentence_count >= 3:
            harder_problems.append(example)
    
    print(f"✓ Filtered GSM8K: {len(harder_problems)} harder problems (from {len(dataset)} total)")
    return harder_problems, "gsm8k_hard"


def format_problem(example, dataset_type):
    """Format a problem based on dataset type."""
    
    if dataset_type == "mrben":
        # MR-Ben format (adjust if needed based on actual format)
        question = example.get('question', example.get('problem', example.get('input', '')))
        answer = example.get('answer', example.get('solution', example.get('output', '')))
    else:  # gsm8k_hard
        question = example['question']
        answer = example['answer']
    
    return question, answer


def generate_trace(model, tokenizer, problem, prompt_template, temperature=0.8, max_new_tokens=1024):
    """Generate a single trace for a problem using specified prompt and temperature."""
    
    prompt = prompt_template.format(problem=problem)
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    trace = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return trace.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate metacognitive traces on MR-Ben")
    parser.add_argument("--num_problems", type=int, default=10, help="Number of problems")
    parser.add_argument("--output_dir", type=str, default="data/metacog_traces/mrben_test")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Load dataset
    dataset, dataset_type = load_dataset_with_fallback()
    
    # Sample problems
    problems = []
    for i in range(min(args.num_problems, len(dataset))):
        problem, answer = format_problem(dataset[i], dataset_type)
        problems.append((problem, answer))
    
    print(f"\n{'='*80}")
    print(f"CONFIGURATION")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_type}")
    print(f"Problems: {len(problems)}")
    print(f"Traces per problem: 4")
    print(f"\nTrace types:")
    print(f"  1. Metacognitive (temp=0.8) - Explicit monitoring + control")
    print(f"  2. Diverse (temp=1.8) - HIGH temp for errors")
    print(f"  3. Reasoning (temp=0.9) - Step-by-step")
    print(f"  4. Direct (temp=0.7) - Minimal")
    print(f"{'='*80}\n")
    
    # Trace configurations - NOTE: temp=1.8 for diverse!
    trace_configs = [
        {
            "name": "metacognitive",
            "prompt": PROMPT_METACOGNITIVE,
            "temperature": 0.8,
            "description": "Explicit metacognition"
        },
        {
            "name": "diverse",
            "prompt": PROMPT_DIRECT,
            "temperature": 1.8,  # ← INCREASED from 1.2 to generate errors!
            "description": "High-temp diverse (error-prone)"
        },
        {
            "name": "reasoning",
            "prompt": PROMPT_REASONING,
            "temperature": 0.9,
            "description": "Step-by-step reasoning"
        },
        {
            "name": "direct",
            "prompt": PROMPT_DIRECT,
            "temperature": 0.7,
            "description": "Direct/minimal"
        },
    ]
    
    all_traces = []
    
    for idx, (problem, answer) in enumerate(tqdm(problems, desc="Generating traces")):
        problem_traces = {
            'problem_id': idx,
            'problem': problem,
            'ground_truth_answer': answer,
            'dataset_type': dataset_type,
            'traces': []
        }
        
        print(f"\n{'='*80}")
        print(f"Problem {idx+1}/{len(problems)}")
        print(f"{'='*80}")
        print(f"Problem: {problem[:150]}...")
        print(f"Ground truth: {answer[:50]}...")
        
        for trace_idx, config in enumerate(trace_configs):
            print(f"\n--- Trace {trace_idx+1}: {config['name']} (temp={config['temperature']}) ---")
            print(f"Description: {config['description']}")
            
            trace = generate_trace(
                model,
                tokenizer,
                problem,
                prompt_template=config['prompt'],
                temperature=config['temperature']
            )
            
            print(f"\nGenerated trace ({len(trace)} chars):")
            print(trace[:300] + "..." if len(trace) > 300 else trace)
            
            problem_traces['traces'].append({
                'trace_id': trace_idx,
                'trace_type': config['name'],
                'temperature': config['temperature'],
                'trace': trace
            })
        
        all_traces.append(problem_traces)
    
    # Save traces
    output_file = output_dir / "traces.json"
    with open(output_file, 'w') as f:
        json.dump(all_traces, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Saved to: {output_file}")
    print(f"Dataset: {dataset_type}")
    print(f"Total problems: {len(all_traces)}")
    print(f"Total traces: {sum(len(p['traces']) for p in all_traces)}")
    print(f"{'='*80}")
    
    # Quick analysis
    print(f"\nQuick check:")
    print(f"- Trace 2 (diverse, temp=1.8) should have some errors")
    print(f"- Compare with other traces to see if errors were caught")
    print(f"\nRun analysis script to check for errors and metacognition quality.")


if __name__ == "__main__":
    main()
