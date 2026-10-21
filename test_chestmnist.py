
from medmnist import ChestMNIST, INFO

print("Checking ChestMNIST dataset...")

# Load each dataset split
train_dataset = ChestMNIST(
    split="train", download=False, size=224, root="data/medmnist"
)

val_dataset = ChestMNIST(
    split="val", download=False, size=224, root="data/medmnist"
)

test_dataset = ChestMNIST(
    split="test", download=False, size=224, root="data/medmnist"
)

# Display dataset sizes
print("\nDataset splits:")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

# Display the 14 labels
print("\nChestMNIST labels:")
for index, name in INFO["chestmnist"]["label"].items():
    print(f"{index}: {name}")

# Inspect one sample
image, labels = train_dataset[0]

print("\nSample inspection:")
print("Image size:", image.size)
print("Image mode:", image.mode)
print("Number of labels:", len(labels))
print("Label values:", labels)
