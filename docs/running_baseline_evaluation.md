# Running Baseline Evaluation

## Overview

This guide provides step-by-step instructions for running the baseline evaluation of Llama-3.1-8B-Instruct on all 5 benchmarks using the proper chat template implementation.

## Prerequisites

1. **Hardware**: H100 GPU (or equivalent with sufficient VRAM)
2. **Environment**: Conda environment `metacog` with all dependencies installed
3. **Model Access**: HuggingFace access token for Llama-3.1-8B-Instruct
4. **Repository**: Latest code from GitHub with chat template implementation

## Step 1: Pull Latest Changes

```bash
cd ~/metacog-reasoning  # or wherever your repo is located
git pull origin main
```

You should see the following changes:
- Updated `scripts/evaluate_baseline.py`
- Updated `src/evaluation/evaluator.py` with chat template support
- New documentation files

## Step 2: Verify Environment

```bash
conda activate metacog
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
```

## Step 3: Run Baseline Evaluation

### Full Evaluation (All 5 Benchmarks)

```bash
python scripts/evaluate_baseline.py \
    --benchmarks gsm8k,mmlu,hellaswag,humaneval,mrben \
    --output_dir results/baseline_instruct \
    --max_samples 1000
```

**Estimated Runtime**: ~20-24 hours on H100

### Individual Benchmark Testing (Recommended First)

Test with a small sample to verify everything works:

```bash
# Test GSM8K (math reasoning)
python scripts/evaluate_baseline.py \
    --benchmarks gsm8k \
    --output_dir results/test_gsm8k \
    --max_samples 10
```

Expected output should show:
- Model loading successfully
- Chat template being applied
- Answers being generated
- Accuracy around 70-80% (even on small sample)

### Benchmark-Specific Commands

```bash
# GSM8K (Grade School Math)
python scripts/evaluate_baseline.py --benchmarks gsm8k --output_dir results/gsm8k --max_samples 1000

# MMLU (Multitask Language Understanding)
python scripts/evaluate_baseline.py --benchmarks mmlu --output_dir results/mmlu --max_samples 1000

# HellaSwag (Commonsense Reasoning)
python scripts/evaluate_baseline.py --benchmarks hellaswag --output_dir results/hellaswag --max_samples 1000

# HumanEval (Code Generation)
python scripts/evaluate_baseline.py --benchmarks humaneval --output_dir results/humaneval --max_samples 164

# MR-Ben (Math Reasoning with Errors)
python scripts/evaluate_baseline.py --benchmarks mrben --output_dir results/mrben --max_samples 1000
```

## Step 4: Monitor Progress

The evaluation script provides progress updates:

```
Loading model from meta-llama/Llama-3.1-8B-Instruct...
Evaluating on gsm8k (test split)...
100%|████████████████████| 1000/1000 [2:30:00<00:00,  9.00s/it]

Results for gsm8k:
  Accuracy: 79.2%
  Total samples: 1000
  Correct: 792
```

### Using tmux for Long-Running Jobs

Since evaluation takes ~24 hours, use tmux to prevent interruption:

```bash
# Start tmux session
tmux new -s baseline_eval

# Run evaluation
python scripts/evaluate_baseline.py \
    --benchmarks gsm8k,mmlu,hellaswag,humaneval,mrben \
    --output_dir results/baseline_instruct \
    --max_samples 1000

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t baseline_eval
```

## Step 5: Check Results

Results are saved in the specified output directory:

```bash
ls -lh results/baseline_instruct/
```

Expected files:
- `gsm8k_results.json` - Detailed results for GSM8K
- `mmlu_results.json` - Detailed results for MMLU
- `hellaswag_results.json` - Detailed results for HellaSwag
- `humaneval_results.json` - Detailed results for HumanEval
- `mrben_results.json` - Detailed results for MR-Ben
- `summary.json` - Overall summary of all benchmarks

### View Summary

```bash
cat results/baseline_instruct/summary.json | python -m json.tool
```

## Step 6: Verify MLflow Tracking

MLflow automatically tracks all experiments:

```bash
# View MLflow UI
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Then open browser to: `http://bmic-h100x2-jo.mdc.musc.edu:5000`

You should see:
- Experiment: `metacog-reasoning`
- Runs for each benchmark
- Metrics: accuracy, total_samples, correct_predictions
- Parameters: model_path, benchmark_name, max_samples

## Expected Baseline Scores

With proper chat template implementation:

| Benchmark | Expected Score | Previous (Generic) | Improvement |
|-----------|---------------|-------------------|-------------|
| **GSM8K** | 75-82% | 61.6% | +13-20% |
| **MMLU** | 65-70% | 52.9% | +12-17% |
| **HellaSwag** | 55-60% | 50.0% | +5-10% |
| **HumanEval** | 70-75% | 15.9% | +54-59% |
| **MR-Ben** | 15-20% | 11.4% | +3.6-8.6% |

**Note**: If scores are significantly lower than expected, verify:
1. Chat template is being applied (check logs)
2. Model loaded correctly (check for warnings)
3. Answer extraction is working (inspect `*_results.json` files)

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size or use model quantization:
```python
# In evaluator.py, modify model loading:
self.model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True,  # Add this line
)
```

### Issue: Slow Generation

**Solution**: Verify GPU is being used:
```bash
nvidia-smi
```

Should show Python process using GPU memory.

### Issue: Low Scores Despite Chat Template

**Possible causes**:
1. Wrong model variant (check it's `-Instruct` not base)
2. Chat template not applied correctly (add debug logging)
3. Answer extraction failing (inspect generated outputs)

**Debug**:
```python
# Add to evaluator.py generate_answer() method:
print(f"Formatted prompt: {prompt[:200]}...")
print(f"Generated response: {full_response[:200]}...")
```

## Next Steps After Baseline

Once baseline evaluation is complete:

1. **Document Results**: Record actual scores in `docs/baseline_results.md`
2. **Analyze Errors**: Review failed predictions in `*_results.json` files
3. **Compare with Literature**: Verify scores align with published benchmarks
4. **Design Self-Play**: Begin implementing metacognitive self-play training
5. **Plan Experiments**: Design training protocol based on Nelson & Narens framework

## Timeline

- **Baseline Evaluation**: 1 day (~24 hours runtime)
- **Results Analysis**: 2-3 hours
- **Documentation**: 1-2 hours
- **Self-Play Design**: 3-5 days
- **Implementation**: 1-2 weeks
- **Training & Evaluation**: 2-3 weeks
- **Paper Writing**: 2-3 weeks
- **Target Submission**: NeurIPS 2026 (May 15-22, 2026)

## Support

If you encounter issues:
1. Check logs in MLflow UI
2. Review error messages in terminal
3. Inspect generated outputs in results files
4. Verify model and tokenizer loaded correctly
5. Test with small sample first (--max_samples 10)

## References

- **Chat Template Documentation**: `docs/chat_template_implementation.md`
- **Model Selection Analysis**: `docs/model_selection_analysis.md`
- **Instruction Tuning Explained**: `docs/instruction_tuning_explained.md`
- **HuggingFace Llama-3.1**: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
