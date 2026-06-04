from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import plot_results
from scipy.signal import savgol_filter, medfilt

csv_paths = [
    "runs/detect/train-14/results.csv",
    "runs/detect/train-15/results.csv",
    "runs/detect/train-18/results.csv"
]
titles = ["Dark dataset", "light dataset", "Dark+light dataset"]
fig, axs = plt.subplots(1, 3)

for i, csv_path in enumerate(csv_paths):

    df = pd.read_csv(csv_path)

    ax = axs[i]
    ax.plot(df["epoch"], df["train/box_loss"])
    ax.plot(df["epoch"], savgol_filter(df["train/box_loss"], window_length=20, polyorder=2))
    ax.set_ylim([0.5, 2.0])
    ax.set_title(titles[i])
    ax.legend(["Box loss", "smoothed loss"])
    #ax.plot(df["epoch"], medfilt(df["train/box_loss"], kernel_size=21))

plt.show()
#plot_results(csv_path)
