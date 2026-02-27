# HW2

Streaming Algorithms Course - Homework 2

## Installation

```bash
poetry install
```

## Usage

### Generate Window Estimations

Run single configuration:
```bash
poetry run python main.py
```

Run all configurations (batch mode with memory tracking):
```bash
poetry run python main.py --run-all
```

### Evaluate Performance

After generating estimations, run comprehensive evaluation:
```bash
python evaluate_performance.py
```

This will generate:
- Precision, Recall, F1-Score, Accuracy metrics
- Confusion matrices
- F1-accuracy vs memory plots
- Recall vs memory plots
- Runtime vs memory plots
- Summary report with best configurations

All results will be saved to `datasets/evaluations/`

### Process Pre-created Sketches

For threshold-based classification:
```bash
python process_precreated_sketches.py
```

## Output Files

- `datasets/training_window_estimations.csv` - Training window estimations
- `datasets/testing_window_estimations.csv` - Testing window estimations
- `datasets/estimations/memory_tracking.csv` - Memory usage by configuration
- `datasets/evaluations/evaluation_results.csv` - Detailed evaluation metrics
- `datasets/evaluations/summary_report.txt` - Best configurations summary
- `datasets/evaluations/*.png` - Visualization plots

## Documentation

See [EVALUATION_README.md](EVALUATION_README.md) for detailed documentation on the evaluation pipeline.

## Reports file
Findings report.docx
