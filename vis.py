from pathlib import Path
from PIL import Image
import cv2
from IPython.display import display
from ultralytics.utils.plotting import Annotator
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


# Classes (annotation)
names = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

# Colors of the classes (annotation)
colors = {
    0: "gray",
    1: "magenta",
    2: "red",
    3: "blue",
    4: "yellow",
    5: "green",
    6: "purple",
    7: "pink",
    8: "orange",
    9: "brown",
}


class Label:
    # definition of labels
    def __init__(self, annotation):
        self.pos = annotation[1:]
        self.cls = annotation[0]


def fig_legend():
    fig, ax = plt.subplots()
    hs = []
    for i in range(len(colors)):
        (h,) = ax.plot([0, 0], [0, 0], color=colors[i], label=names[i])
        hs.append(h)
    ax.legend(handles=hs)
    plt.show()


# fig_legend()


def read_labels(label_path):
    """Reads the labels and positions for a single annotation file

    Args:
        label_path (str): the file path of the annotation file

    Returns:
        List[Label]: Returns the a list with labels: class/position
    """
    with open(label_path, encoding="utf-8") as file:
        labels = []
        for row in [x.split(" ") for x in file.read().strip().splitlines()]:
            # print(row)
            # All position values are normalized
            cls, x_center, y_center, w_norm, h_norm = map(float, row[:])
            cls = int(cls)
            labels.append(Label([cls, x_center, y_center, w_norm, h_norm]))
    return labels


def show_cv2_img(img):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
    display(Image.fromarray(im))


def show_labeled_img(img_path, label_path=None, label_text=False):
    img = cv2.imread(img_path)
    fig, ax = plt.subplots()
    if not label_path is None:
        labels = read_labels(label_path)
        for label in labels:
            pos = np.array(
                [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
            ) * np.array(label.pos)
            x_center, y_center, w, h = pos
            x = x_center - w / 2
            y = y_center - h / 2
            pos = [x, y, w, h]
            ax.plot(
                [x, x, x + w, x + w, x],
                [y, y + h, y + h, y, y],
                colors[label.cls],
                label=names[label.cls],
            )
            # annotator = Annotator(img, line_width=3)
            # annotator.box_label(pos, names[label.cls], (255, 0, 0))
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.legend()
    ax.imshow(im)
    # fig.show()
    return fig
    # display(Image.fromarray(im))
