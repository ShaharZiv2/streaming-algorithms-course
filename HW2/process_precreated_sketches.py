import pandas as pd

data = pd.read_csv('datasets/window_estimations.csv')

x = data.groupby('Attack').mean()
pass
