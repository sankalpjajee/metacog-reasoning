#!/usr/bin/env python3
"""Compare multiple MLflow experiments."""

import argparse
import mlflow
import pandas as pd
from tabulate import tabulate


def compare_experiments(experiment_names=None, metric="accuracy"):
    """
    Compare experiments from MLflow tracking.
    
    Args:
        experiment_names: List of experiment names to compare (None = all)
        metric: Metric to compare
    """
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Get all experiments
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    
    if experiment_names:
        experiments = [e for e in experiments if e.name in experiment_names]
    
    results = []
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id)
        
        for run in runs:
            results.append({
                'Experiment': exp.name,
                'Run': run.info.run_name,
                'Model': run.data.params.get('model', 'N/A'),
                'Benchmark': run.data.params.get('benchmark', 'N/A'),
                'Accuracy': run.data.metrics.get(metric, 0),
                'Samples': run.data.metrics.get('num_samples', 0),
                'Timestamp': run.info.start_time,
            })
    
    if not results:
        print("No experiments found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('Accuracy', ascending=False)
    
    # Print table
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80 + "\n")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    summary = df.groupby('Benchmark')['Accuracy'].agg(['mean', 'std', 'min', 'max'])
    print(tabulate(summary, headers='keys', tablefmt='grid'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare MLflow experiments")
    parser.add_argument(
        '--experiments',
        nargs='+',
        help='Experiment names to compare (default: all)'
    )
    parser.add_argument(
        '--metric',
        default='accuracy',
        help='Metric to compare (default: accuracy)'
    )
    
    args = parser.parse_args()
    compare_experiments(args.experiments, args.metric)
