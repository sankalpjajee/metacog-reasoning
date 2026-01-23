#!/usr/bin/env python3
"""
Generate metacognitive traces for self-play training.
This script generates multiple reasoning traces per problem with explicit metacognitive markers.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Metacognitive prompt template
METACOGNITIVE_PROMPT = """You are a careful problem solver. For each problem, think step-by-step and explicitly show your metacognitive reasoning:

1. State your confidence level after each reasoning step
2. Check for potential errors in your reasoning  
3. Verify your final answer

Use these markers:
- [Confidence: High/Medium/Low] after steps you're certain/uncertain about
- [Check: ...] when verifying your reasoning
- [Verify: ...] when double-checking your final answer

Problem: {problem}

Think carefully and show your metacognitive reasoning:"""


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
    temperature=0.8
):
    """Generate multiple traces for each problem."""
    
    all_traces = []
    
    for idx, (problem, answer) in enumerate(tqdm(problems, desc="Generating traces")):
        problem_traces = {
            'problem_id': idx,
            'problem': problem,
            'ground_truth_answer': answer,
            'traces': []
        }
        
        # Generate multiple traces with different temperatures for diversity
        temperatures = [temperature] * num_traces_per_problem
        
        for trace_idx, temp in enumerate(temperatures):
            print(f"\n{'='*80}")
            print(f"Problem {idx+1}/{len(problems)}, Trace {trace_idx+1}/{num_traces_per_problem}")
            print(f"{'='*80}")
            print(f"Problem: {problem[:100]}...")
            
            trace = generate_trace(model, tokenizer, problem, temperature=temp)
            
            print(f"\nGenerated trace (temp={temp}):")
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
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
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
    print(f"Temperature: {args.temperature}")
    print(f"Total traces to generate: {args.num_problems * args.num_traces}")
    
    # Generate traces
    all_traces = generate_traces_for_problems(
        model,
        tokenizer,
        problems,
        num_traces_per_problem=args.num_traces,
        temperature=args.temperature
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
    
    # Check for metacognitive markers
    markers = ['[Confidence:', '[Check:', '[Verify:', 'confident', 'uncertain']
    traces_with_markers = 0
    for problem_data in all_traces:
        for trace_data in problem_data['traces']:
            trace = trace_data['trace'].lower()
            if any(marker.lower() in trace for marker in markers):
                traces_with_markers += 1
                break
    
    print(f"- Traces with metacognitive markers: {traces_with_markers}/{total_traces} ({traces_with_markers/total_traces*100:.1f}%)")


if __name__ == "__main__":
    main()
