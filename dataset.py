from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import os
import torch


class UpscalerDataset(Dataset):
    def __init__(self, dataset_folder: str, device: torch.DeviceObjType) -> None:
        super().__init__()
        self.imagenames = os.listdir(os.path.join(dataset_folder, "Upscale"))
        self.basepath = dataset_folder
        self.device = device

    def __len__(self):
        return len(self.imagenames)

    def __getitem__(self, idx):
        x = read_image(os.path.join(os.path.join(
            self.basepath, "Downscale"), self.imagenames[idx])).to(self.device)
        y = read_image(os.path.join(os.path.join(
            self.basepath, "Upscale"), self.imagenames[idx])).to(self.device)
        return x/255, y/255
