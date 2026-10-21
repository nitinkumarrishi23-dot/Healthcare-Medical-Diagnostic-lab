import numpy as np
from medmnist import ChestMNIST, INFO
dataset = ChestMNIST(
    split="train",
    download=False,
    size=224,
    root="data/medmnist"
)
# Each row represents an image and each column represents a disease.
labels = dataset.labels
print("Training images:", len(dataset))
print("Label matrix shape:", labels.shape)
print("\nDisease label distribution:\n")
for index, name in INFO["chestmnist"]["label"].items():
    positive = int(np.sum(labels[:, int(index)] == 1))
    negative = len(dataset) - positive
    percentage = positive / len(dataset) * 100

    print(
        f"{name:18} | "
        f"Positive: {positive:6} | "
        f"Negative: {negative:6} | "
        f"Positive rate: {percentage:5.2f}%"
    )
