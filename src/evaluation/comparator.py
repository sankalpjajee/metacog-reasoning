"""Utilities for comparing multiple models."""

import json
import os
from typing import List, Dict
from pathlib import Path


class ModelComparator:
    """Compare evaluation results from multiple models."""
    
    def __init__(self, result_dirs: List[str], model_names: List[str] = None):
        """
        Initialize comparator.
        
        Args:
            result_dirs: List of directories containing evaluation results
            model_names: Optional list of model names (defaults to directory names)
        """
        self.result_dirs = result_dirs
        self.model_names = model_names or [Path(d).name for d in result_dirs]
        self.results = self._load_results()
    
    def _load_results(self) -> Dict[str, Dict]:
        """Load results from all directories."""
        all_results = {}
        
        for model_name, result_dir in zip(self.model_names, self.result_dirs):
            summary_file = os.path.join(result_dir, "summary.json")
            
            if not os.path.exists(summary_file):
                print(f"Warning: Summary file not found for {model_name}: {summary_file}")
                continue
            
            with open(summary_file, 'r') as f:
                all_results[model_name] = json.load(f)
        
        return all_results
    
    def generate_comparison_table(self) -> str:
        """Generate a markdown comparison table."""
        if not self.results:
            return "No results to compare."
        
        # Get all benchmarks
        benchmarks = set()
        for results in self.results.values():
            benchmarks.update(results['benchmarks'].keys())
        benchmarks = sorted(benchmarks)
        
        # Create table header
        lines = ["# Model Comparison Results\n"]
        lines.append("| Benchmark | " + " | ".join(self.model_names) + " | Best |")
        lines.append("|:----------|" + "|".join([":----------" for _ in self.model_names]) + "|:-----|")
        
        # Add rows for each benchmark
        for benchmark in benchmarks:
            row = [benchmark.upper()]
            accuracies = []
            
            for model_name in self.model_names:
                if model_name in self.results and benchmark in self.results[model_name]['benchmarks']:
                    acc = self.results[model_name]['benchmarks'][benchmark]['accuracy']
                    accuracies.append(acc)
                    row.append(f"{acc:.1%}")
                else:
                    accuracies.append(0.0)
                    row.append("N/A")
            
            # Mark best result
            best_acc = max(accuracies)
            best_model = self.model_names[accuracies.index(best_acc)]
            row.append(f"**{best_model}**")
            
            lines.append("| " + " | ".join(row) + " |")
        
        # Add average row
        avg_row = ["**Average**"]
        avg_accuracies = []
        for model_name in self.model_names:
            if model_name in self.results:
                avg_acc = self.results[model_name].get('average_accuracy', 0.0)
                avg_accuracies.append(avg_acc)
                avg_row.append(f"**{avg_acc:.1%}**")
            else:
                avg_accuracies.append(0.0)
                avg_row.append("N/A")
        
        best_avg = max(avg_accuracies)
        best_avg_model = self.model_names[avg_accuracies.index(best_avg)]
        avg_row.append(f"**{best_avg_model}**")
        
        lines.append("| " + " | ".join(avg_row) + " |")
        
        return "\n".join(lines)
    
    def compute_improvements(self, baseline_name: str) -> Dict:
        """
        Compute improvements relative to a baseline.
        
        Args:
            baseline_name: Name of the baseline model
        
        Returns:
            Dictionary with improvement percentages
        """
        if baseline_name not in self.results:
            raise ValueError(f"Baseline {baseline_name} not found in results")
        
        baseline = self.results[baseline_name]
        improvements = {}
        
        for model_name in self.model_names:
            if model_name == baseline_name or model_name not in self.results:
                continue
            
            model_results = self.results[model_name]
            model_improvements = {}
            
            for benchmark in baseline['benchmarks']:
                if benchmark not in model_results['benchmarks']:
                    continue
                
                baseline_acc = baseline['benchmarks'][benchmark]['accuracy']
                model_acc = model_results['benchmarks'][benchmark]['accuracy']
                
                if baseline_acc > 0:
                    improvement = ((model_acc - baseline_acc) / baseline_acc) * 100
                    model_improvements[benchmark] = improvement
            
            # Compute average improvement
            if model_improvements:
                avg_improvement = sum(model_improvements.values()) / len(model_improvements)
                model_improvements['average'] = avg_improvement
            
            improvements[f"{model_name}_vs_{baseline_name}"] = model_improvements
        
        return improvements
    
    def generate_comparison_report(self, baseline_name: str = None) -> Dict:
        """
        Generate a comprehensive comparison report.
        
        Args:
            baseline_name: Optional baseline model name for computing improvements
        
        Returns:
            Dictionary with comparison data
        """
        report = {
            'models': self.model_names,
            'benchmarks': {},
        }
        
        # Collect results for each benchmark
        for model_name, results in self.results.items():
            for benchmark, bench_results in results['benchmarks'].items():
                if benchmark not in report['benchmarks']:
                    report['benchmarks'][benchmark] = {}
                report['benchmarks'][benchmark][model_name] = bench_results['accuracy']
        
        # Add improvements if baseline is specified
        if baseline_name:
            report['improvements'] = self.compute_improvements(baseline_name)
        
        return report
    
    def save_comparison(self, output_dir: str, baseline_name: str = None):
        """
        Save comparison results to files.
        
        Args:
            output_dir: Directory to save comparison files
            baseline_name: Optional baseline for computing improvements
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save markdown table
        table = self.generate_comparison_table()
        table_file = os.path.join(output_dir, "comparison_table.md")
        with open(table_file, 'w') as f:
            f.write(table)
        print(f"Comparison table saved to {table_file}")
        
        # Generate and save JSON report
        report = self.generate_comparison_report(baseline_name)
        report_file = os.path.join(output_dir, "comparison_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Comparison report saved to {report_file}")
        
        # Print table to console
        print("\n" + table)
        
        # Print improvements if available
        if baseline_name and 'improvements' in report:
            print(f"\n## Improvements vs {baseline_name}\n")
            for comparison, improvements in report['improvements'].items():
                print(f"\n### {comparison}")
                for benchmark, improvement in improvements.items():
                    if benchmark != 'average':
                        print(f"  {benchmark}: {improvement:+.1f}%")
                if 'average' in improvements:
                    print(f"  **Average: {improvements['average']:+.1f}%**")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare model evaluation results")
    parser.add_argument(
        "--result_dirs",
        type=str,
        required=True,
        help="Comma-separated list of result directories"
    )
    parser.add_argument(
        "--model_names",
        type=str,
        default=None,
        help="Comma-separated list of model names (optional)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline model name for computing improvements"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for comparison files"
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    result_dirs = [d.strip() for d in args.result_dirs.split(',')]
    model_names = None
    if args.model_names:
        model_names = [n.strip() for n in args.model_names.split(',')]
    
    # Create comparator
    comparator = ModelComparator(result_dirs, model_names)
    
    # Save comparison
    comparator.save_comparison(args.output_dir, args.baseline)
