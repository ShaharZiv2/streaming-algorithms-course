import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from matplotlib import pyplot as plt


TRAINING_DATA_PATH = 'datasets/full_run/training_window_estimations.csv'
BASELINE_DATA_PATH = 'datasets/full_run/baseline_training_window_estimations.csv'
TESTING_DATA_PATH = 'datasets/full_run/testing_window_estimations.csv'

NORMALIZABLE_COLUMNS = ['SrcIPF0',
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
    data[NORMALIZABLE_COLUMNS] = data[NORMALIZABLE_COLUMNS].div(data['Events_F1'], axis=0)


def train_ml_model(data:pd.DataFrame):
    # Train decision tree
    y = data['Attack']
    X = data.drop('Attack', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Save model (optional)
    import joblib
    joblib.dump(clf, 'models/decision_tree_classifier.pkl')


def plot_thresholds(grouped_data: pd.DataFrame):
    feature_cols = grouped_data.columns
    fig, axes = plt.subplots(7, 3, figsize=(20, 30))

    for col, ax in zip(feature_cols, axes.flatten()):
        grouped_data[col].plot(kind='bar', ax=ax, color=['green', 'red'])
        ax.set_title(f'{col} by Attack Label')
        ax.set_xlabel('Attack (0 = Normal, 1 = Attack)')
        ax.set_ylabel(f'Mean {col}')
        ax.set_xticklabels(['Normal', 'Attack'], rotation=0)

    plt.tight_layout()
    plt.show()


def get_thresholds(grouped_data: pd.DataFrame):
    grouped_t = grouped_data.T
    grouped_t['mean'] = grouped_t.mean(axis=1)
    grouped_t['diff'] = (grouped_t[0] - grouped_t[1]).abs()
    significant_diffs = grouped_t[grouped_t['diff'] / grouped_t['mean'] > 0.3]
    significant_diffs['above_is_attack'] = significant_diffs['mean'] > significant_diffs[0]
    significant_diffs.rename(columns={'mean': 'threshold'}, inplace=True)

    return significant_diffs[['threshold', 'above_is_attack']].T


def prepare_data_for_rule_based(data: pd.DataFrame, thresholds: pd.DataFrame):
    filtered_data = data[thresholds.columns]
    threshold_values = thresholds.loc['threshold', filtered_data.columns].values
    above_is_attack = thresholds.loc['above_is_attack', filtered_data.columns].values

    return filtered_data, threshold_values, above_is_attack


def rule_based_classify(filtered_data, threshold_values, above_is_attack, minimum_thresholds):
    pass_from_below = (filtered_data > threshold_values) & above_is_attack
    pass_from_above = (filtered_data <= threshold_values) & (~above_is_attack)
    passed = pass_from_below | pass_from_above
    estimated_attack = passed.sum(axis=1) >= minimum_thresholds
    return estimated_attack


def get_optimal_thresholds_count(data: pd.DataFrame, thresholds: pd.DataFrame, plot: bool = True):
    filtered_data, threshold_values, above_is_attack = prepare_data_for_rule_based(data, thresholds)
    correct_labeling = []
    current_max = 0
    best_threshold_count = 0
    for i in range(1, len(thresholds.columns) + 1):
        classifications = rule_based_classify(filtered_data, threshold_values, above_is_attack, i)
        correct_labeling_count = (classifications == data['Attack'].astype(bool)).sum()
        if correct_labeling_count > current_max:
            current_max = correct_labeling_count
            best_threshold_count = i

        correct_labeling.append(correct_labeling_count)

    if plot:
        plt.bar(range(1, len(thresholds.columns) + 1), correct_labeling)
        plt.title('Correct labeling count by thresholds passed')
        plt.show()

    return best_threshold_count, current_max


def process_threshold_classification(training_data: pd.DataFrame, testing_data: pd.DataFrame, plot: bool = True):
    training_grouped = training_data.groupby('Attack').mean()
    if plot:
        plot_thresholds(training_grouped)

    thresholds = get_thresholds(training_grouped)
    best_threshold_count, success_count = get_optimal_thresholds_count(training_data, thresholds)
    print(f'Using our threshold methodology, we successfully classified {success_count / len(training_data) * 100:.3f}% of the training windows')
    print(f'This was achieved by classifying True if at least {best_threshold_count} were passed')
    print('We will now classify the testing set according to the training')

    filtered_data, threshold_values, above_is_attack = prepare_data_for_rule_based(testing_data, thresholds)
    classifications = rule_based_classify(filtered_data, threshold_values, above_is_attack, best_threshold_count)
    correct_labeling_count = (classifications == testing_data['Attack'].astype(bool)).sum()
    print(f'For the testing data, we successfully classified {correct_labeling_count / len(testing_data) * 100:.3f}% of the testing windows')


def plot_accuracies(sketch_windows: pd.DataFrame, baseline_windows: pd.DataFrame):
    ratio_df = (sketch_windows / baseline_windows).drop(columns='Attack')
    feature_cols = ratio_df.columns

    num_cols = 3
    num_rows = (len(feature_cols) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))

    for col, ax in zip(feature_cols, axes.flatten()):
        print('Plotting accuracy for', col)
        ax.hist(ratio_df[col].dropna(), bins=50, edgecolor='black')
        ax.set_title(f'{col} (Sketch / Baseline)')
        ax.set_xlabel('Ratio')
        ax.set_ylabel('Frequency')
        ax.axvline(x=1.0, color='r', linestyle='--', label='Perfect accuracy')
        ax.legend()

    # Hide unused subplots
    print('Removing unused subplots')
    for ax in axes.flatten()[len(feature_cols):]:
        ax.set_visible(False)

    print('Showing')
    plt.tight_layout()
    plt.show()


def main():
    training_windows = pd.read_csv(TRAINING_DATA_PATH)
    testing_windows = pd.read_csv(TESTING_DATA_PATH)
    baseline_training_window = pd.read_csv(BASELINE_DATA_PATH)
    plot_accuracies(training_windows, baseline_training_window)
    normalize_data(training_windows)
    normalize_data(testing_windows)
    process_threshold_classification(training_windows, testing_windows)
    # train_ml_model(data)



if __name__ == '__main__':
    main()
