# MLflow Experiment Tracking

This project uses [MLflow](https://mlflow.org/) to track experiments, metrics, and model performance across all benchmarks.

## Quick Start

### 1. Run Evaluation (Automatic Tracking)

MLflow tracking is enabled by default:

```bash
python scripts/evaluate_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k,mmlu,hellaswag,humaneval,mrben \
    --output_dir data/results/baseline
```

### 2. View Results in MLflow UI

```bash
bash scripts/mlflow_ui.sh
```

Then open http://localhost:5000 in your browser.

### 3. Compare Experiments

```bash
python scripts/compare_experiments.py
```

## What Gets Tracked

### Parameters
- `model`: Model name/path
- `benchmark`: Benchmark name
- `split`: Dataset split (train/test/validation)
- `max_new_tokens`: Generation length
- `device`: cuda/cpu
- `max_samples`: Number of samples (if limited)

### Metrics
- `accuracy`: Overall accuracy
- `num_correct`: Number of correct predictions
- `num_incorrect`: Number of incorrect predictions
- `num_samples`: Total samples evaluated
- `accuracy_{category}`: Per-category accuracy
- `accuracy_level_{difficulty}`: Per-difficulty accuracy

### Artifacts
- `{benchmark}_results.json`: Detailed predictions
- `summary.json`: Evaluation summary

## MLflow UI Features

### Experiments View
- Compare multiple runs side-by-side
- Filter by parameters or metrics
- Sort by any column
- Search runs by name

### Run Details
- View all parameters and metrics
- Download artifacts (results files)
- See run timeline and duration
- Compare with other runs

### Metrics Visualization
- Plot metrics over time
- Compare metrics across runs
- Export charts

## Advanced Usage

### Disable MLflow Tracking

```python
from src.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    use_mlflow=False  # Disable tracking
)
```

### Custom Experiment Name

```python
evaluator = ModelEvaluator(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    experiment_name="my-custom-experiment"
)
```

### Query Runs Programmatically

```python
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

# Get all runs from an experiment
experiment = client.get_experiment_by_name("metacog-reasoning")
runs = client.search_runs(experiment.experiment_id)

for run in runs:
    print(f"Run: {run.info.run_name}")
    print(f"Accuracy: {run.data.metrics['accuracy']}")
    print(f"Model: {run.data.params['model']}")
```

### Export Results to CSV

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name("metacog-reasoning")
runs = client.search_runs(experiment.experiment_id)

data = []
for run in runs:
    data.append({
        'run_name': run.info.run_name,
        'model': run.data.params.get('model'),
        'benchmark': run.data.params.get('benchmark'),
        'accuracy': run.data.metrics.get('accuracy'),
        'num_samples': run.data.metrics.get('num_samples'),
    })

df = pd.DataFrame(data)
df.to_csv('experiment_results.csv', index=False)
```

## Directory Structure

```
metacog-reasoning/
├── mlruns/                    # MLflow tracking data
│   ├── 0/                     # Default experiment
│   ├── 1/                     # metacog-reasoning experiment
│   │   ├── meta.yaml          # Experiment metadata
│   │   └── {run_id}/          # Individual runs
│   │       ├── meta.yaml      # Run metadata
│   │       ├── metrics/       # Metric values
│   │       ├── params/        # Parameter values
│   │       └── artifacts/     # Result files
│   └── .trash/                # Deleted runs
└── data/results/              # JSON result files (also in MLflow)
```

## Tips

1. **Run Names**: Automatically generated as `{benchmark}_{split}_{timestamp}`
2. **Artifacts**: All result files are logged to MLflow and can be downloaded from UI
3. **Comparison**: Use the "Compare" button in UI to see side-by-side metrics
4. **Search**: Use the search bar to filter runs by parameters (e.g., `params.model = "Llama"`)
5. **Export**: Download any artifact or export metrics to CSV from the UI

## Troubleshooting

### MLflow UI not starting
```bash
# Check if port is in use
lsof -i :5000

# Use different port
bash scripts/mlflow_ui.sh 5001
```

### Cannot find experiments
```bash
# Check tracking URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Should output: file:./mlruns
```

### Runs not appearing
- Make sure you're in the project root directory
- Check that `mlruns/` directory exists
- Verify evaluation ran with `use_mlflow=True` (default)

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
