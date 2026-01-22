import pandas as pd

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


if __name__ == '__main__':
    main()
