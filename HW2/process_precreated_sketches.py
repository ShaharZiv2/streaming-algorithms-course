import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = 'datasets/window_estimations.csv'
NORMALIZABLE_COLUMNS = ['SrcIP_F0',
                        'SrcIP_F2',
                        'DstIP_F0',
                        'DstIP_F2',
                        'DstPortF0',
                        'DstPortF2',
                        ]


def normalize_data(data: pd.DataFrame):
    data[NORMALIZABLE_COLUMNS] = data[NORMALIZABLE_COLUMNS].div(data['NumEventsF1'], axis=0)


def main():
    data = pd.read_csv(DATA_PATH)
    normalize_data(data)
    data = data.groupby('Attack').mean()
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


if __name__ == '__main__':
    main()
