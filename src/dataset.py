
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from medmnist import ChestMNIST


class ChestMNISTDataset(Dataset):
    def __init__(self, split="train", root="data/medmnist"):
        self.dataset = ChestMNIST(
            split=split,
            download=False,
            size=224,
            root=root
        )

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, labels = self.dataset[index]

        image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels
