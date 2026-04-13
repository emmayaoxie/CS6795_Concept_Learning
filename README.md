# CS6795 Concept Learning Simulator

This repository contains the final code and experiment outputs for a CS6795 project on **top-down and bottom-up concept learning**. The code implements a small, interpretable proof of concept that compares several learning strategies on an ambiguous concept domain: deciding whether an item counts as **soup**.

## What the code does

The project models four learning approaches:
- **Top-down rule learner**: parses a written concept definition into weighted feature constraints.
- **Bottom-up incremental learner**: updates a contrastive representation from labeled examples.
- **Case-based reasoner**: stores exemplars and classifies by similarity to previous examples.
- **Hybrid learner**: combines rule-based prior knowledge, bottom-up abstraction, and exemplar memory.

The repository also includes an experiment runner that:
- evaluates the learners under three instructional orders (`canonical_first`, `boundary_first`, and `negatives_first`)
- exports JSON and CSV results
- generates figures used in the final report
- writes additional diagnostics such as error analysis and hybrid explanations

## Repository contents

### Main source files
- `concept_learning_simulator.py`  
  Core data structures, dataset, learning algorithms, evaluation logic, and experiment orchestration.

- `run_experiments.py`  
  Runs the full experiment pipeline and writes results, diagnostics, and plots.

### Generated outputs
- `week8_results.json` — full experiment trace with step-by-step results
- `step_metrics.csv` — metrics for each model at each learning step
- `final_summary.csv` — final performance summary by sequence and model
- `error_analysis.csv` — final misclassified examples
- `hybrid_explanations.csv` — hybrid model score breakdowns and exception adjustments
- `code_insights.json` — summary of code-level findings used in the report
- `representation_history_boundary_first.csv` — bottom-up feature contrast history

### Figures
- `hybrid_sequence_comparison.png`
- `accuracy_canonical_first.png`
- `accuracy_boundary_first.png`
- `accuracy_negatives_first.png`
- `representation_evolution.png`

These figures are the plots referenced in the final report.

## Requirements

This code was written in Python and uses only the standard library plus `matplotlib`.

Recommended:
- Python 3.9+
- `matplotlib`

Install dependency:

```bash
python -m pip install matplotlib
```

## How to run the code

Run the full experiment pipeline from the repository root:

```bash
python run_experiments.py
```

This writes results and figures into the current directory by default.

### Optional arguments

Use a different output directory:

```bash
python run_experiments.py --output_dir results
```

Override the default natural-language definition used by the top-down learner:

```bash
python run_experiments.py --definition "Soup is a savory liquid food served as part of a meal."
```

## Expected outputs

After running the script, you should see:
- JSON and CSV summaries of model performance
- plots comparing learning curves across models and instructional orders
- diagnostic files for the hybrid learner and error cases
- a generated `README.md` summary if that helper is invoked by the runner

## Notes

- This repository is a proof of concept, not a production system.
- The feature space and dataset are intentionally small so that the learning behavior remains interpretable.
- The main purpose of the code is to support the cognitive science analysis in the final report.
