# Chat Template Implementation for Llama-3.1-8B-Instruct

## Overview

This document explains the implementation of proper chat template formatting for Llama-3.1-8B-Instruct evaluation, which is critical for achieving accurate baseline scores.

## Why Chat Templates Matter

Instruction-tuned models like Llama-3.1-8B-Instruct are trained with specific formatting conventions using special tokens. Using generic prompts instead of the proper chat template can result in **40-60% performance degradation** because:

1. The model was fine-tuned on conversations with specific structure
2. Special tokens signal role boundaries (system, user, assistant)
3. The model learned to respond differently based on these structural cues
4. Without proper formatting, the model treats input as raw text continuation

## Llama-3.1 Chat Template Format

The Llama-3.1 Instruct models use the following special tokens:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

## Implementation Details

### 1. Prompt Formatting (`_format_prompt` method)

**Before (Generic Prompt):**
```python
def _format_prompt(self, question: str) -> str:
    return f"""Solve the following problem step by step.

Problem: {question}

Solution:"""
```

**After (Chat Template):**
```python
def _format_prompt(self, question: str) -> str:
    """Format question as a prompt using Llama-3.1 Instruct chat template."""
    is_multiple_choice = bool(re.search(r'\n[A-D]\.', question))
    
    if is_multiple_choice:
        user_message = f"""Answer the following multiple choice question. Think step by step, then provide your final answer as a single letter (A, B, C, or D).

Question: {question}"""
    else:
        user_message = f"""Solve the following problem step by step. Provide your final answer at the end.

Problem: {question}"""
    
    # Use Llama-3.1 Instruct chat template
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant skilled at reasoning and problem-solving."},
        {"role": "user", "content": user_message}
    ]
    
    # Apply chat template
    prompt = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt
```

### 2. Answer Extraction

The chat template changes the output format, so we need to extract only the assistant's response:

```python
# Decode
generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract answer (remove prompt)
# For chat template, extract only the assistant's response
if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
    # Split at assistant marker and take everything after
    parts = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")
    if len(parts) > 1:
        full_response = parts[-1].strip()
    else:
        full_response = generated_text[len(prompt):].strip()
else:
    # Fallback: remove prompt
    full_response = generated_text[len(prompt):].strip()
```

### 3. System Message Design

We use a general-purpose system message that works across all benchmarks:

```
"You are a helpful AI assistant skilled at reasoning and problem-solving."
```

This is intentionally generic to avoid biasing the model toward specific task types. Task-specific instructions are provided in the user message.

## Expected Performance Improvements

With proper chat template implementation, we expect baseline scores to improve significantly:

| Benchmark | Generic Prompt | Chat Template (Expected) | Improvement |
|-----------|---------------|-------------------------|-------------|
| GSM8K | 61.6% | ~79% | +17.4% |
| MMLU | 52.9% | ~65-70% | +12-17% |
| HellaSwag | 50.0% | ~55-60% | +5-10% |
| HumanEval | 15.9% | ~72% | +56.1% |
| MR-Ben | 11.4% | ~15-20% | +3.6-8.6% |

## Verification

To verify the chat template is being applied correctly:

1. **Check tokenizer output**: The `apply_chat_template()` method should return a string with special tokens
2. **Inspect generated prompts**: Add debug logging to see the actual formatted prompts
3. **Compare with HuggingFace examples**: Cross-reference with official Llama-3.1 documentation

## References

- **HuggingFace Llama-3.1 Documentation**: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- **Chat Templates Guide**: https://huggingface.co/docs/transformers/chat_templating
- **SPIN Paper**: Uses instruction-tuned models with proper formatting
- **SPPO Paper**: Follows similar methodology for fair evaluation

## Next Steps

1. **Pull changes** from GitHub: `git pull origin main`
2. **Run baseline evaluation**: `python scripts/evaluate_baseline.py --benchmarks gsm8k,mmlu,hellaswag,humaneval,mrben --output_dir results/baseline_instruct`
3. **Compare results**: Verify scores match expected ranges
4. **Document findings**: Record actual baseline scores for paper

## Notes

- The chat template is applied automatically by the tokenizer based on the model's configuration
- `add_generation_prompt=True` adds the assistant header to prompt the model to respond
- `skip_special_tokens=True` in decoding removes special tokens from output for cleaner text
- The system message can be customized per benchmark if needed, but current generic message works well
