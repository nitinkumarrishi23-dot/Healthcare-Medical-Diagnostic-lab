
from torch.utils.data import DataLoader
from src.dataset import ChestMNISTDataset

dataset = ChestMNISTDataset(split="train")

print("Total training images:", len(dataset))

image, labels = dataset[0]

print("Single image shape:", image.shape)
print("Single label shape:", labels.shape)
print("Label values:", labels)

loader = DataLoader(dataset, batch_size=4, shuffle=True)
images, labels = next(iter(loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
