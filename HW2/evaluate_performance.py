import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from matplotlib import pyplot as plt
import seaborn as sns
import glob
import time


NORMALIZABLE_COLUMNS = [
    'SrcIPF0',
    'DstIPF0',
    'DstPortF0',
    'SrcIPF2',
    'DstIPF2',
    'DstPortF2',
    'SrcIP_MaxCount',
    'DstIP_MaxCount',
    'DstPort_MaxCount',
]


def normalize_data(data: pd.DataFrame):
    """Normalize features by dividing by the number of events"""
    data[NORMALIZABLE_COLUMNS] = data[NORMALIZABLE_COLUMNS].div(data['Events_F1'], axis=0)


def load_configuration_data(cms_width, ams_r, iteration):
    """Load training and testing data for a specific configuration"""
    training_file = f'datasets/estimations/trng_{cms_width}_{ams_r}_{iteration}.csv'
    testing_file = f'datasets/estimations/tstng_{cms_width}_{ams_r}_{iteration}.csv'

    if not os.path.exists(training_file) or not os.path.exists(testing_file):
        return None, None

    training_data = pd.read_csv(training_file)
    testing_data = pd.read_csv(testing_file)

    normalize_data(training_data)
    normalize_data(testing_data)

    return training_data, testing_data


def train_and_evaluate_model(training_data, testing_data):
    """
    Train a decision tree model and evaluate it on testing data.
    Returns metrics dictionary.
    """
    start_time = time.time()

    # Prepare training data
    y_train = training_data['Attack']
    X_train = training_data.drop('Attack', axis=1)

    # Prepare testing data
    y_test = testing_data['Attack']
    X_test = testing_data.drop('Attack', axis=1)

    # Train model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    runtime = time.time() - start_time

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'runtime': runtime
    }

    return metrics, y_test, y_pred


def evaluate_all_configurations():
    """Evaluate all precreated configurations and collect metrics"""
    results = []

    cms_widths = [2048, 8192]
    ams_rs = [16, 64, 256]

    for cms_width in cms_widths:
        for ams_r in ams_rs:
            for iteration in range(25):
                print(f'Evaluating: cms_width={cms_width}, ams_r={ams_r}, iteration={iteration}')

                training_data, testing_data = load_configuration_data(cms_width, ams_r, iteration)

                if training_data is None:
                    print(f'  Skipping - files not found')
                    continue

                try:
                    metrics, y_test, y_pred = train_and_evaluate_model(training_data, testing_data)

                    # Store results
                    result = {
                        'cms_width': cms_width,
                        'ams_r': ams_r,
                        'iteration': iteration,
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1': metrics['f1'],
                        'runtime': metrics['runtime'],
                        'confusion_matrix': metrics['confusion_matrix'].tolist(),
                        'tn': metrics['confusion_matrix'][0, 0],
                        'fp': metrics['confusion_matrix'][0, 1],
                        'fn': metrics['confusion_matrix'][1, 0],
                        'tp': metrics['confusion_matrix'][1, 1],
                    }

                    results.append(result)
                    print(f'  Accuracy: {metrics["accuracy"]:.4f}, F1: {metrics["f1"]:.4f}, '
                          f'Recall: {metrics["recall"]:.4f}, Runtime: {metrics["runtime"]:.4f}s')

                except Exception as e:
                    print(f'  Error: {e}')
                    continue

    return pd.DataFrame(results)


def merge_with_memory_data(results_df):
    """Merge evaluation results with memory tracking data"""
    memory_file = 'datasets/estimations/memory_tracking.csv'

    if not os.path.exists(memory_file):
        print(f"Warning: Memory tracking file not found at {memory_file}")
        return results_df

    memory_df = pd.read_csv(memory_file)

    # Merge on configuration parameters
    merged_df = results_df.merge(
        memory_df,
        on=['cms_width', 'ams_r', 'iteration'],
        how='left'
    )

    return merged_df


def plot_confusion_matrix_average(results_df, save_path='datasets/evaluations'):
    """Plot average confusion matrices for each configuration"""
    os.makedirs(save_path, exist_ok=True)

    configs = results_df.groupby(['cms_width', 'ams_r'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, ((cms_width, ams_r), group) in enumerate(configs):
        # Average confusion matrix across iterations
        avg_tn = group['tn'].mean()
        avg_fp = group['fp'].mean()
        avg_fn = group['fn'].mean()
        avg_tp = group['tp'].mean()

        avg_cm = np.array([[avg_tn, avg_fp], [avg_fn, avg_tp]])

        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'],
                   ax=axes[idx])
        axes[idx].set_title(f'CMS Width: {cms_width}, AMS r: {ams_r}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f'Saved confusion matrices to {save_path}/confusion_matrices.png')
    plt.close()


def plot_f1_vs_memory(results_df, save_path='datasets/evaluations'):
    """Plot F1 score vs memory usage"""
    os.makedirs(save_path, exist_ok=True)

    if 'testing_total_memory_mb' not in results_df.columns:
        print("Warning: Memory data not available, skipping F1 vs Memory plot")
        return

    # Group by configuration and calculate means
    grouped = results_df.groupby(['cms_width', 'ams_r']).agg({
        'f1': 'mean',
        'testing_total_memory_mb': 'mean'
    }).reset_index()

    plt.figure(figsize=(10, 6))

    for cms_width in grouped['cms_width'].unique():
        subset = grouped[grouped['cms_width'] == cms_width]
        plt.plot(subset['testing_total_memory_mb'], subset['f1'],
                marker='o', markersize=10, label=f'CMS Width: {cms_width}',
                linewidth=2)

        # Annotate points with ams_r values
        for _, row in subset.iterrows():
            plt.annotate(f"r={row['ams_r']}",
                        (row['testing_total_memory_mb'], row['f1']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.xlabel('Memory Usage (MB)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs Memory Usage', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/f1_vs_memory.png', dpi=300, bbox_inches='tight')
    print(f'Saved F1 vs Memory plot to {save_path}/f1_vs_memory.png')
    plt.close()


def plot_recall_vs_memory(results_df, save_path='datasets/evaluations'):
    """Plot Recall vs memory usage"""
    os.makedirs(save_path, exist_ok=True)

    if 'testing_total_memory_mb' not in results_df.columns:
        print("Warning: Memory data not available, skipping Recall vs Memory plot")
        return

    # Group by configuration and calculate means
    grouped = results_df.groupby(['cms_width', 'ams_r']).agg({
        'recall': 'mean',
        'testing_total_memory_mb': 'mean'
    }).reset_index()

    plt.figure(figsize=(10, 6))

    for cms_width in grouped['cms_width'].unique():
        subset = grouped[grouped['cms_width'] == cms_width]
        plt.plot(subset['testing_total_memory_mb'], subset['recall'],
                marker='s', markersize=10, label=f'CMS Width: {cms_width}',
                linewidth=2)

        # Annotate points with ams_r values
        for _, row in subset.iterrows():
            plt.annotate(f"r={row['ams_r']}",
                        (row['testing_total_memory_mb'], row['recall']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.xlabel('Memory Usage (MB)', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Recall vs Memory Usage', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/recall_vs_memory.png', dpi=300, bbox_inches='tight')
    print(f'Saved Recall vs Memory plot to {save_path}/recall_vs_memory.png')
    plt.close()


def plot_runtime_vs_memory(results_df, save_path='datasets/evaluations'):
    """Plot Runtime vs memory usage"""
    os.makedirs(save_path, exist_ok=True)

    if 'testing_total_memory_mb' not in results_df.columns:
        print("Warning: Memory data not available, skipping Runtime vs Memory plot")
        return

    # Group by configuration and calculate means
    grouped = results_df.groupby(['cms_width', 'ams_r']).agg({
        'runtime': 'mean',
        'testing_total_memory_mb': 'mean'
    }).reset_index()

    plt.figure(figsize=(10, 6))

    for cms_width in grouped['cms_width'].unique():
        subset = grouped[grouped['cms_width'] == cms_width]
        plt.plot(subset['testing_total_memory_mb'], subset['runtime'],
                marker='^', markersize=10, label=f'CMS Width: {cms_width}',
                linewidth=2)

        # Annotate points with ams_r values
        for _, row in subset.iterrows():
            plt.annotate(f"r={row['ams_r']}",
                        (row['testing_total_memory_mb'], row['runtime']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.xlabel('Memory Usage (MB)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Runtime vs Memory Usage', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/runtime_vs_memory.png', dpi=300, bbox_inches='tight')
    print(f'Saved Runtime vs Memory plot to {save_path}/runtime_vs_memory.png')
    plt.close()


def plot_all_metrics_comparison(results_df, save_path='datasets/evaluations'):
    """Plot comparison of all metrics across configurations"""
    os.makedirs(save_path, exist_ok=True)

    # Group by configuration
    grouped = results_df.groupby(['cms_width', 'ams_r']).agg({
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped['config'] = grouped['cms_width'].astype(str) + '_' + grouped['ams_r'].astype(str)

    # Create bar plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        x = np.arange(len(grouped))
        width = 0.6

        bars = ax.bar(x, grouped[f'{metric}_mean'], width,
                     yerr=grouped[f'{metric}_std'], capsize=5,
                     alpha=0.7, edgecolor='black')

        # Color bars by cms_width
        colors = ['skyblue' if w == 2048 else 'lightcoral'
                 for w in grouped['cms_width']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Configuration (CMS_Width_AMS_r)', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f'{title} by Configuration', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped['config'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', label='CMS Width: 2048'),
        Patch(facecolor='lightcoral', edgecolor='black', label='CMS Width: 8192')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{save_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f'Saved metrics comparison to {save_path}/metrics_comparison.png')
    plt.close()


def generate_summary_report(results_df, save_path='datasets/evaluations'):
    """Generate a text summary report"""
    os.makedirs(save_path, exist_ok=True)

    # Group by configuration
    grouped = results_df.groupby(['cms_width', 'ams_r']).agg({
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'runtime': ['mean', 'std']
    })

    if 'testing_total_memory_mb' in results_df.columns:
        memory_grouped = results_df.groupby(['cms_width', 'ams_r'])['testing_total_memory_mb'].mean()
        grouped['memory_mb'] = memory_grouped

    # Save to file
    report_path = f'{save_path}/summary_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STREAMING ALGORITHMS EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("Performance Metrics by Configuration\n")
        f.write("-" * 80 + "\n\n")

        for (cms_width, ams_r), row in grouped.iterrows():
            f.write(f"Configuration: CMS Width = {cms_width}, AMS r = {ams_r}\n")
            f.write(f"  Accuracy:  {row[('accuracy', 'mean')]:.4f} ± {row[('accuracy', 'std')]:.4f}\n")
            f.write(f"  Precision: {row[('precision', 'mean')]:.4f} ± {row[('precision', 'std')]:.4f}\n")
            f.write(f"  Recall:    {row[('recall', 'mean')]:.4f} ± {row[('recall', 'std')]:.4f}\n")
            f.write(f"  F1 Score:  {row[('f1', 'mean')]:.4f} ± {row[('f1', 'std')]:.4f}\n")
            f.write(f"  Runtime:   {row[('runtime', 'mean')]:.4f} ± {row[('runtime', 'std')]:.4f} seconds\n")
            if 'memory_mb' in grouped.columns:
                memory_val = grouped.loc[(cms_width, ams_r), 'memory_mb']
                f.write(f"  Memory:    {memory_val:.2f} MB\n")
            f.write("\n")

        # Find best configurations
        best_f1_idx = grouped[('f1', 'mean')].idxmax()
        best_recall_idx = grouped[('recall', 'mean')].idxmax()
        best_precision_idx = grouped[('precision', 'mean')].idxmax()

        f.write("-" * 80 + "\n")
        f.write("Best Configurations:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best F1 Score:  CMS Width = {best_f1_idx[0]}, AMS r = {best_f1_idx[1]} "
                f"(F1 = {grouped.loc[best_f1_idx, ('f1', 'mean')]:.4f})\n")
        f.write(f"Best Recall:    CMS Width = {best_recall_idx[0]}, AMS r = {best_recall_idx[1]} "
                f"(Recall = {grouped.loc[best_recall_idx, ('recall', 'mean')]:.4f})\n")
        f.write(f"Best Precision: CMS Width = {best_precision_idx[0]}, AMS r = {best_precision_idx[1]} "
                f"(Precision = {grouped.loc[best_precision_idx, ('precision', 'mean')]:.4f})\n")

    print(f'Saved summary report to {report_path}')


def main():
    """Main evaluation pipeline"""
    print("Starting evaluation of all configurations...")
    print("=" * 80)

    # Evaluate all configurations
    results_df = evaluate_all_configurations()

    if results_df.empty:
        print("No results to process. Make sure data files exist.")
        return

    print(f"\nProcessed {len(results_df)} configurations")

    # Merge with memory data
    results_df = merge_with_memory_data(results_df)

    # Save results
    os.makedirs('datasets/evaluations', exist_ok=True)
    results_df.to_csv('datasets/evaluations/evaluation_results.csv', index=False)
    print(f"\nSaved detailed results to datasets/evaluations/evaluation_results.csv")

    # Generate all visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix_average(results_df)
    plot_all_metrics_comparison(results_df)
    plot_f1_vs_memory(results_df)
    plot_recall_vs_memory(results_df)
    plot_runtime_vs_memory(results_df)

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(results_df)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("All results saved to datasets/evaluations/")


if __name__ == '__main__':
    main()
