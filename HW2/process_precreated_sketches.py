import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('datasets/window_estimations.csv')

x = data.groupby('Attack').mean()
pass