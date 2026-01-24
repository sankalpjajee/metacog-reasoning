# Changelog

All notable changes to the metacog-reasoning project.

## [2026-01-20] - Chat Template Implementation

### Added
- **Chat template support** for Llama-3.1-8B-Instruct evaluation
  - Implemented `tokenizer.apply_chat_template()` in `_format_prompt()` method
  - Added system message: "You are a helpful AI assistant skilled at reasoning and problem-solving"
  - Proper formatting with special tokens for instruction-tuned model

- **Improved answer extraction** for chat template output
  - Extracts only assistant's response from formatted output
  - Handles `<|start_header_id|>assistant<|end_header_id|>` marker
  - Fallback mechanism for compatibility

- **Comprehensive documentation**
  - `docs/chat_template_implementation.md` - Technical details of implementation
  - `docs/running_baseline_evaluation.md` - Step-by-step guide for running evaluation
  - Expected performance improvements documented

### Changed
- **Updated `src/evaluation/evaluator.py`**
  - Modified `_format_prompt()` to use chat template instead of generic prompts
  - Updated `generate_answer()` to extract assistant responses correctly
  - Maintained backward compatibility with fallback extraction

- **Updated `scripts/evaluate_baseline.py`**
  - Documentation reflects Instruct model usage
  - Default model set to `meta-llama/Llama-3.1-8B-Instruct`

### Why These Changes Matter

**Problem**: Using generic prompts with instruction-tuned models causes 40-60% performance degradation because the model was fine-tuned on specific conversation formats with special tokens.

**Solution**: Implement proper chat template formatting to match the model's training distribution.

**Expected Impact**:
- GSM8K: 61.6% → ~79% (+17.4%)
- MMLU: 52.9% → ~65-70% (+12-17%)
- HellaSwag: 50.0% → ~55-60% (+5-10%)
- HumanEval: 15.9% → ~72% (+56.1%)
- MR-Ben: 11.4% → ~15-20% (+3.6-8.6%)

### Methodology Alignment

These changes align our evaluation methodology with established practices in the field:
- **SPIN (Self-Play Fine-Tuning)**: Uses instruction-tuned models with proper formatting
- **SPPO (Self-Play Preference Optimization)**: Follows similar evaluation protocols
- **Standard Practice**: All major papers use chat templates for instruction-tuned models

### Next Steps

1. **Pull changes** from GitHub: `git pull origin main`
2. **Run baseline evaluation** (~24 hours on H100)
3. **Verify scores** match expected ranges
4. **Document actual results** for paper
5. **Begin self-play training** design and implementation

### Technical Details

**Chat Template Format** (Llama-3.1):
```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant skilled at reasoning and problem-solving.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

**Implementation**:
```python
messages = [
    {"role": "system", "content": "You are a helpful AI assistant skilled at reasoning and problem-solving."},
    {"role": "user", "content": user_message}
]

prompt = self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

### Files Modified
- `scripts/evaluate_baseline.py`
- `src/evaluation/evaluator.py`

### Files Added
- `docs/chat_template_implementation.md`
- `docs/running_baseline_evaluation.md`
- `CHANGELOG.md`

### Commits
- `c29d19f` - Update evaluation to use Llama-3.1-8B-Instruct chat template
- `457d3c2` - Add documentation for chat template implementation
- `1445328` - Add comprehensive guide for running baseline evaluation

---

## [2026-01-19] - Initial Setup

### Added
- Repository structure and initial codebase
- Evaluation pipeline for 5 benchmarks (GSM8K, MMLU, HellaSwag, HumanEval, MR-Ben)
- MLflow integration for experiment tracking
- Benchmark loaders and evaluation metrics
- Documentation on model selection and instruction tuning

### Completed
- Initial baseline evaluation with generic prompts
- Literature review (SPIN, SPPO papers)
- Model selection decision (Llama-3.1-8B-Instruct)
- Syntax fixes in evaluator.py
- MR-Ben prompt improvements

### Results (Generic Prompts - Not Representative)
- GSM8K: 61.6%
- MMLU: 52.9%
- HellaSwag: 50.0%
- HumanEval: 15.9%
- MR-Ben: 11.4%

**Note**: These scores are artificially low due to improper prompt formatting. See 2026-01-20 changes for proper evaluation.
