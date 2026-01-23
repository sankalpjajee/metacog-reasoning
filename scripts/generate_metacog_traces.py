#!/usr/bin/env python3
"""
Generate metacognitive traces for self-play training.

Goal: Prove that metacognition improves accuracy by helping models catch and correct errors.

Strategy: Generate 4 diverse traces per problem:
  1. Correct + strong metacognition (catches errors, verifies work)
  2. Diverse/error-prone (higher temp, may have errors)
  3. Correct + some reasoning (baseline good response)
  4. Direct/minimal (efficient but no checking)
"""

import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Different prompts for different trace types
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


def format_gsm8k_problem(example):
    """Format a GSM8K problem."""
    return example['question'], example['answer']


def classify_problem_difficulty(problem):
    """
    Classify problem difficulty based on heuristics.
    Simple: Single-step arithmetic
    Medium: Multi-step but straightforward
    Hard: Multi-step with potential error traps
    """
    words = problem.lower().split()
    
    # Count mathematical operations
    operations = sum(1 for word in words if word in ['+', '-', '*', '/', 'plus', 'minus', 'times', 'divided'])
    
    # Count steps (rough estimate based on sentence count)
    sentences = problem.count('.') + problem.count('?')
    
    if sentences <= 1 and operations <= 1:
        return "simple"
    elif sentences >= 3 or operations >= 3:
        return "hard"
    else:
        return "medium"


def generate_trace(model, tokenizer, problem, prompt_template, temperature=0.8, max_new_tokens=1024):
    """Generate a single trace for a problem using specified prompt and temperature."""
    
    # Format the prompt
    prompt = prompt_template.format(problem=problem)
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    trace = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return trace.strip()


def generate_traces_for_problems(
    model, 
    tokenizer, 
    problems, 
    num_traces_per_problem=4,
):
    """
    Generate multiple traces for each problem with different characteristics.
    
    Trace types:
    1. Metacognitive (explicit monitoring and control) - temp 0.8
    2. High-temp diverse (may have errors) - temp 1.2
    3. Reasoning (step-by-step) - temp 0.9
    4. Direct (minimal) - temp 0.7
    """
    
    all_traces = []
    
    # Configuration for each trace type
    trace_configs = [
        {
            "name": "metacognitive",
            "prompt": PROMPT_METACOGNITIVE,
            "temperature": 0.8,
            "description": "Explicit metacognition (monitoring + control)"
        },
        {
            "name": "diverse",
            "prompt": PROMPT_DIRECT,
            "temperature": 1.2,
            "description": "High-temp diverse (may have errors)"
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
    
    for idx, (problem, answer) in enumerate(tqdm(problems, desc="Generating traces")):
        # Classify problem difficulty
        difficulty = classify_problem_difficulty(problem)
        
        problem_traces = {
            'problem_id': idx,
            'problem': problem,
            'ground_truth_answer': answer,
            'difficulty': difficulty,
            'traces': []
        }
        
        print(f"\n{'='*80}")
        print(f"Problem {idx+1}/{len(problems)} (Difficulty: {difficulty})")
        print(f"{'='*80}")
        print(f"Problem: {problem[:150]}...")
        print(f"Ground truth: {answer[:50]}...")
        
        for trace_idx in range(min(num_traces_per_problem, len(trace_configs))):
            config = trace_configs[trace_idx]
            
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
            print(trace[:400] + "..." if len(trace) > 400 else trace)
            
            problem_traces['traces'].append({
                'trace_id': trace_idx,
                'trace_type': config['name'],
                'temperature': config['temperature'],
                'trace': trace
            })
        
        all_traces.append(problem_traces)
    
    return all_traces


def analyze_traces(all_traces):
    """Analyze generated traces for quality metrics."""
    
    total_traces = sum(len(p['traces']) for p in all_traces)
    
    # Count traces with metacognitive markers
    metacog_markers = ['[confidence:', '[check:', '[verify:', 'wait', 'let me check', 'let me verify']
    monitoring_markers = ['confidence:', 'certain', 'uncertain', 'sure', 'unsure']
    control_markers = ['check:', 'verify:', 'wait', 'error', 'mistake', 'correct']
    
    traces_with_metacog = 0
    traces_with_monitoring = 0
    traces_with_control = 0
    
    for problem_data in all_traces:
        for trace_data in problem_data['traces']:
            trace_lower = trace_data['trace'].lower()
            
            if any(marker in trace_lower for marker in metacog_markers):
                traces_with_metacog += 1
            
            if any(marker in trace_lower for marker in monitoring_markers):
                traces_with_monitoring += 1
            
            if any(marker in trace_lower for marker in control_markers):
                traces_with_control += 1
    
    return {
        'total_traces': total_traces,
        'traces_with_metacog': traces_with_metacog,
        'traces_with_monitoring': traces_with_monitoring,
        'traces_with_control': traces_with_control,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate metacognitive traces")
    parser.add_argument(
        "--num_problems",
        type=int,
        default=10,
        help="Number of problems to generate traces for"
    )
    parser.add_argument(
        "--num_traces",
        type=int,
        default=4,
        help="Number of traces per problem"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/metacog_traces/prototype",
        help="Output directory for traces"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for generation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Load GSM8K dataset
    print("\nLoading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # Sample problems
    problems = []
    for i in range(args.num_problems):
        problem, answer = format_gsm8k_problem(dataset[i])
        problems.append((problem, answer))
    
    print(f"\n{'='*80}")
    print(f"TRACE GENERATION CONFIGURATION")
    print(f"{'='*80}")
    print(f"Problems: {args.num_problems}")
    print(f"Traces per problem: {args.num_traces}")
    print(f"Total traces: {args.num_problems * args.num_traces}")
    print(f"\nTrace types:")
    print(f"  1. Metacognitive (temp=0.8) - Explicit monitoring + control")
    print(f"  2. Diverse (temp=1.2) - High temp for potential errors")
    print(f"  3. Reasoning (temp=0.9) - Step-by-step reasoning")
    print(f"  4. Direct (temp=0.7) - Minimal/efficient")
    print(f"{'='*80}")
    
    # Generate traces
    all_traces = generate_traces_for_problems(
        model,
        tokenizer,
        problems,
        num_traces_per_problem=args.num_traces,
    )
    
    # Save traces
    output_file = output_dir / "traces.json"
    with open(output_file, 'w') as f:
        json.dump(all_traces, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Traces saved to: {output_file}")
    print(f"Total problems: {len(all_traces)}")
    print(f"Total traces: {sum(len(p['traces']) for p in all_traces)}")
    
    # Analyze traces
    print(f"\n{'='*80}")
    print(f"TRACE ANALYSIS")
    print(f"{'='*80}")
    
    stats = analyze_traces(all_traces)
    
    print(f"Total traces: {stats['total_traces']}")
    print(f"Traces with metacognitive markers: {stats['traces_with_metacog']} ({stats['traces_with_metacog']/stats['total_traces']*100:.1f}%)")
    print(f"Traces with monitoring: {stats['traces_with_monitoring']} ({stats['traces_with_monitoring']/stats['total_traces']*100:.1f}%)")
    print(f"Traces with control: {stats['traces_with_control']} ({stats['traces_with_control']/stats['total_traces']*100:.1f}%)")
    
    # Count by difficulty
    difficulty_counts = {}
    for problem_data in all_traces:
        diff = problem_data['difficulty']
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    
    print(f"\nProblem difficulty distribution:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count} problems")
    
    print(f"\n{'='*80}")
    print(f"Average trace length: {sum(len(t['trace']) for p in all_traces for t in p['traces']) / stats['total_traces']:.0f} characters")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
