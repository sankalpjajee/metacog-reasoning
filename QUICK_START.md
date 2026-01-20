# Quick Start Guide - Baseline Evaluation

## What Changed

The evaluation code now uses **proper Llama-3.1-8B-Instruct chat template** instead of generic prompts. This is critical for accurate baseline scores.

## Quick Commands

### 1. Pull Latest Changes
```bash
cd ~/metacog-reasoning
git pull origin main
```

### 2. Test Small Sample (Recommended First)
```bash
conda activate metacog
python scripts/evaluate_baseline.py \
    --benchmarks gsm8k \
    --output_dir results/test \
    --max_samples 10
```

### 3. Run Full Baseline (~24 hours)
```bash
# Using tmux to prevent interruption
tmux new -s baseline

python scripts/evaluate_baseline.py \
    --benchmarks gsm8k,mmlu,hellaswag,humaneval,mrben \
    --output_dir results/baseline_instruct \
    --max_samples 1000

# Detach: Ctrl+B then D
# Reattach: tmux attach -t baseline
```

### 4. Check Results
```bash
cat results/baseline_instruct/summary.json | python -m json.tool
```

## Expected Scores

| Benchmark | Previous | Expected Now | Improvement |
|-----------|----------|--------------|-------------|
| GSM8K | 61.6% | ~79% | +17.4% |
| MMLU | 52.9% | ~65-70% | +12-17% |
| HellaSwag | 50.0% | ~55-60% | +5-10% |
| HumanEval | 15.9% | ~72% | +56.1% |
| MR-Ben | 11.4% | ~15-20% | +3.6-8.6% |

## What Was Changed

1. **`src/evaluation/evaluator.py`**:
   - `_format_prompt()` now uses `tokenizer.apply_chat_template()`
   - Added system message for instruction-tuned model
   - Fixed answer extraction for chat format

2. **`scripts/evaluate_baseline.py`**:
   - Updated documentation to reflect Instruct model

## Documentation

- **Technical Details**: `docs/chat_template_implementation.md`
- **Full Guide**: `docs/running_baseline_evaluation.md`
- **Changes**: `CHANGELOG.md`

## Troubleshooting

**Low scores?**
- Verify model is `Llama-3.1-8B-Instruct` (not base)
- Check chat template is applied (look for special tokens in logs)
- Test with small sample first

**CUDA OOM?**
- Reduce batch size
- Use 8-bit quantization (see docs)

**Need help?**
- Check MLflow UI: `mlflow ui --port 5000`
- Review detailed results in `*_results.json` files
- See full documentation in `docs/` directory

## Next Steps After Baseline

1. Document actual baseline scores
2. Analyze error patterns
3. Design metacognitive self-play training
4. Implement training protocol
5. Target: NeurIPS 2026 (May 15-22, 2026)
