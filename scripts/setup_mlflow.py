#!/usr/bin/env python3
"""Setup MLflow experiment tracking."""

import mlflow
import os

def setup_mlflow():
    """Initialize MLflow experiment."""
    
    # Set tracking URI to local directory
    mlflow_dir = os.path.expanduser("~/metacog-reasoning/mlruns")
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    
    # Create or get experiment
    experiment_name = "metacog-reasoning-baseline"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f"{mlflow_dir}/artifacts"
            )
            print(f"✓ Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✓ Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
        
        # Set as default experiment
        mlflow.set_experiment(experiment_name)
        
        print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment name: {experiment_name}")
        print(f"Experiment ID: {experiment_id}")
        print("\n✓ MLflow setup complete!")
        
        return experiment_id
        
    except Exception as e:
        print(f"Error setting up MLflow: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    setup_mlflow()
