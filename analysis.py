from ultralytics import YOLO
import pandas as pd

df1 = pd.read_csv("runs/detect/train-9/results.csv")
df2 = pd.read_csv("runs/detect/train-8/results.csv")

print(df1)
df2["epoch"] = df2["epoch"] + 100
df2.index = df2.index + 100
print(df2)

df = pd.concat([df1, df2])
print(df)

from ultralytics.utils.plotting import plot_results

csv_path = "results_with_pt.csv"
df.to_csv(csv_path)
# Specify the path to your CSV file
plot_results(csv_path)

