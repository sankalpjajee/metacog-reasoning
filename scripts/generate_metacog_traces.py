#!/usr/bin/env python3
"""
Generate metacognitive traces for self-play training.
This script generates multiple reasoning traces per problem with varying levels of metacognition.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Flexible metacognitive prompt template
METACOGNITIVE_PROMPT = """You are a helpful problem solver. Solve this problem accurately.

For complex problems, think carefully and verify your work.
For simple problems, direct answers are fine.
If you're uncertain, express your uncertainty.

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
    # Simple heuristics (can be improved)
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


def generate_trace(model, tokenizer, problem, temperature=0.8, max_new_tokens=1024):
    """Generate a single metacognitive trace for a problem."""
    
    # Format the prompt
    prompt = METACOGNITIVE_PROMPT.format(problem=problem)
    
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
    """Generate multiple traces for each problem with varying temperatures."""
    
    all_traces = []
    
    # Use different temperatures to get natural diversity
    # Lower temp → more likely correct, less metacognition
    # Higher temp → more diverse, some errors, more metacognition
    temperatures = [0.7, 0.8, 0.9, 1.0]
    
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
        
        for trace_idx in range(num_traces_per_problem):
            temp = temperatures[trace_idx % len(temperatures)]
            
            print(f"\n{'='*80}")
            print(f"Problem {idx+1}/{len(problems)}, Trace {trace_idx+1}/{num_traces_per_problem}")
            print(f"Difficulty: {difficulty}, Temperature: {temp}")
            print(f"{'='*80}")
            print(f"Problem: {problem[:100]}...")
            
            trace = generate_trace(model, tokenizer, problem, temperature=temp)
            
            print(f"\nGenerated trace:")
            print(trace[:500] + "..." if len(trace) > 500 else trace)
            
            problem_traces['traces'].append({
                'trace_id': trace_idx,
                'temperature': temp,
                'trace': trace
            })
        
        all_traces.append(problem_traces)
    
    return all_traces


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
    
    print(f"\nGenerating {args.num_traces} traces for {args.num_problems} problems...")
    print(f"Using temperatures: [0.7, 0.8, 0.9, 1.0] for diversity")
    print(f"Total traces to generate: {args.num_problems * args.num_traces}")
    
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
    print(f"Traces saved to: {output_file}")
    print(f"Total problems: {len(all_traces)}")
    print(f"Total traces: {len(all_traces) * args.num_traces}")
    print(f"{'='*80}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    total_traces = sum(len(p['traces']) for p in all_traces)
    avg_length = sum(len(t['trace']) for p in all_traces for t in p['traces']) / total_traces
    print(f"- Average trace length: {avg_length:.0f} characters")
    
    # Count by difficulty
    difficulty_counts = {}
    for problem_data in all_traces:
        diff = problem_data['difficulty']
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    
    print(f"\nProblem difficulty distribution:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count} problems")
    
    # Check for metacognitive markers
    markers = ['[Confidence:', '[Check:', '[Verify:', 'confident', 'uncertain', 'let me', 'wait']
    traces_with_markers = 0
    for problem_data in all_traces:
        for trace_data in problem_data['traces']:
            trace = trace_data['trace'].lower()
            if any(marker.lower() in trace for marker in markers):
                traces_with_markers += 1
    
    print(f"\nTraces with metacognitive markers: {traces_with_markers}/{total_traces} ({traces_with_markers/total_traces*100:.1f}%)")


if __name__ == "__main__":
    main()
