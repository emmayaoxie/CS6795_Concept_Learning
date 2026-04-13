"""Run experiments and export diagnostics for the concept learning proof of concept."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from concept_learning_simulator import FEATURES, run_all_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--definition", type=str, default=None, help="Override the default concept definition.")
    return parser.parse_args()


def write_json(results: dict, results_dir: Path) -> None:
    with open(results_dir / "week8_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def write_step_metrics_csv(results: dict, results_dir: Path) -> None:
    output_path = results_dir / "step_metrics.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sequence",
            "step",
            "example_name",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "misclassified_examples",
        ])
        for sequence_name, sequence_result in results.items():
            for trace in sequence_result["trace"]:
                step = trace["step"]
                example_name = trace["example_name"]
                for model_name, metrics in trace["evaluations"].items():
                    writer.writerow([
                        sequence_name,
                        step,
                        example_name,
                        model_name,
                        metrics["accuracy"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1"],
                        ", ".join(item["name"] for item in metrics["misclassified"]),
                    ])


def write_final_summary_csv(results: dict, results_dir: Path) -> None:
    output_path = results_dir / "final_summary.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sequence",
            "model",
            "final_accuracy",
            "final_precision",
            "final_recall",
            "final_f1",
            "mean_accuracy_over_steps",
            "final_misclassified_examples",
        ])
        for sequence_name, sequence_result in results.items():
            model_names = list(sequence_result["trace"][-1]["evaluations"].keys())
            for model_name in model_names:
                final_metrics = sequence_result["trace"][-1]["evaluations"][model_name]
                mean_accuracy = mean(
                    trace["evaluations"][model_name]["accuracy"]
                    for trace in sequence_result["trace"]
                )
                writer.writerow([
                    sequence_name,
                    model_name,
                    final_metrics["accuracy"],
                    final_metrics["precision"],
                    final_metrics["recall"],
                    final_metrics["f1"],
                    round(mean_accuracy, 3),
                    ", ".join(item["name"] for item in final_metrics["misclassified"]),
                ])


def write_representation_history_csv(results: dict, results_dir: Path, sequence_name: str = "boundary_first") -> None:
    output_path = results_dir / f"representation_history_{sequence_name}.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "example_name", *FEATURES])
        for trace in results[sequence_name]["trace"]:
            contrast = trace["representation"]["contrast"]
            writer.writerow([
                trace["step"],
                trace["example_name"],
                *[contrast[feature] for feature in FEATURES],
            ])


def write_error_analysis_csv(results: dict, results_dir: Path) -> None:
    output_path = results_dir / "error_analysis.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "model", "example", "gold", "prediction", "score"])
        for sequence_name, sequence_result in results.items():
            final_evaluations = sequence_result["trace"][-1]["evaluations"]
            for model_name, metrics in final_evaluations.items():
                for item in metrics["misclassified"]:
                    writer.writerow([
                        sequence_name,
                        model_name,
                        item["name"],
                        item["gold"],
                        item["prediction"],
                        item["score"],
                    ])


def write_hybrid_explanations_csv(results: dict, results_dir: Path) -> None:
    output_path = results_dir / "hybrid_explanations.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sequence",
            "example",
            "gold",
            "prediction",
            "final_score",
            "rule_weight",
            "feature_weight",
            "case_weight",
            "rule_score",
            "feature_score",
            "case_score",
            "exception_adjustment",
            "nearest_positive",
            "nearest_negative",
        ])
        for sequence_name, sequence_result in results.items():
            hybrid_predictions = sequence_result["final_predictions"]["hybrid"]
            for example_name, payload in hybrid_predictions.items():
                explanation = payload["explanation"]
                writer.writerow([
                    sequence_name,
                    example_name,
                    payload["label"],
                    payload["prediction"],
                    explanation["final_score"],
                    explanation["rule_weight"],
                    explanation["feature_weight"],
                    explanation["case_weight"],
                    explanation["rule_score"],
                    explanation["feature_score"],
                    explanation["case_score"],
                    explanation["exception_adjustment"],
                    explanation["nearest_positive"],
                    explanation["nearest_negative"],
                ])


def write_code_insights(results: dict, results_dir: Path) -> None:
    summary = {}
    for sequence_name, sequence_result in results.items():
        summary[sequence_name] = sequence_result["code_insights"]
    with open(results_dir / "code_insights.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def make_accuracy_plot(results: dict, figures_dir: Path) -> None:
    for sequence_name, sequence_result in results.items():
        plt.figure(figsize=(7.2, 4.2))
        steps = [trace["step"] for trace in sequence_result["trace"]]
        for model_name in ["top_down", "bottom_up", "case_based", "hybrid"]:
            accuracies = [trace["evaluations"][model_name]["accuracy"] for trace in sequence_result["trace"]]
            plt.plot(steps, accuracies, marker="o", linewidth=2, label=model_name.replace("_", " ").title())
        plt.xticks(steps)
        plt.ylim(0.3, 1.0)
        plt.xlabel("Learning step")
        plt.ylabel("Held-out accuracy")
        plt.title(f"Accuracy over time: {sequence_name.replace('_', ' ').title()}")
        plt.grid(alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(figures_dir / f"accuracy_{sequence_name}.png", dpi=220)
        plt.close()


def make_hybrid_sequence_plot(results: dict, figures_dir: Path) -> None:
    plt.figure(figsize=(7.0, 4.0))
    for sequence_name, sequence_result in results.items():
        steps = [trace["step"] for trace in sequence_result["trace"]]
        accuracies = [trace["evaluations"]["hybrid"]["accuracy"] for trace in sequence_result["trace"]]
        plt.plot(steps, accuracies, marker="o", linewidth=2, label=sequence_name.replace("_", " ").title())
    plt.xticks(range(1, 9))
    plt.ylim(0.3, 1.0)
    plt.xlabel("Learning step")
    plt.ylabel("Hybrid held-out accuracy")
    plt.title("Hybrid learner under different instructional orders")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figures_dir / "hybrid_sequence_comparison.png", dpi=220)
    plt.close()


def make_representation_plot(results: dict, figures_dir: Path, sequence_name: str = "boundary_first") -> None:
    sequence_result = results[sequence_name]
    plt.figure(figsize=(7.2, 4.6))
    steps = [trace["step"] for trace in sequence_result["trace"]]
    for feature in ["savory", "sweet", "drink_like", "spoonable", "meal_like", "served_hot"]:
        values = [trace["representation"]["contrast"][feature] for trace in sequence_result["trace"]]
        plt.plot(steps, values, marker="o", linewidth=2, label=feature.replace("_", " "))
    plt.axhline(0, linewidth=1)
    plt.xticks(steps)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("Learning step")
    plt.ylabel("Positive-negative contrast")
    plt.title("Representation evolution in the bottom-up learner")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(figures_dir / "representation_evolution.png", dpi=220)
    plt.close()


def write_readme_summary(results: dict, output_dir: Path) -> None:
    lines = []
    lines.append("# Week 8 experiment summary")
    lines.append("")
    lines.append("This folder contains the final code and experiment outputs for the CS6795 concept-learning simulator.")
    lines.append("")
    lines.append("## Main code-level takeaways")
    lines.append("- The top-down learner is static after parsing the written definition, so its accuracy curve stays flat across steps.")
    lines.append("- The bottom-up learner changes only when new positive/negative contrast appears in the feature counts.")
    lines.append("- The hybrid learner starts with stronger rule influence, then shifts weight toward feature abstraction and exemplar memory after three training examples.")
    lines.append("- The only persistent hybrid error on the held-out set is lentil_stew, suggesting that the current feature vocabulary lacks a strong thickness/viscosity cue.")
    lines.append("")
    lines.append("## Final held-out accuracies by sequence")
    lines.append("")
    for sequence_name, sequence_result in results.items():
        lines.append(f"### {sequence_name}")
        final_evals = sequence_result["trace"][-1]["evaluations"]
        for model_name in ["top_down", "bottom_up", "case_based", "hybrid"]:
            metrics = final_evals[model_name]
            lines.append(
                f"- {model_name}: accuracy={metrics['accuracy']}, precision={metrics['precision']}, recall={metrics['recall']}, f1={metrics['f1']}"
            )
        lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = run_all_sequences(definition=args.definition) if args.definition else run_all_sequences()
    write_json(results, results_dir)
    write_step_metrics_csv(results, results_dir)
    write_final_summary_csv(results, results_dir)
    write_representation_history_csv(results, results_dir)
    write_error_analysis_csv(results, results_dir)
    write_hybrid_explanations_csv(results, results_dir)
    write_code_insights(results, results_dir)
    make_accuracy_plot(results, figures_dir)
    make_hybrid_sequence_plot(results, figures_dir)
    make_representation_plot(results, figures_dir)
    write_readme_summary(results, output_dir)
    print(f"Wrote results to: {results_dir}")
    print(f"Wrote figures to: {figures_dir}")


if __name__ == "__main__":
    main()
