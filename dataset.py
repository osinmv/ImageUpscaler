from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import os
import torch


class UpscalerDataset(Dataset):
    def __init__(self, dataset_folder: str, device: torch.DeviceObjType, size: Tuple[int, int] = (1200, 1600)) -> None:
        super().__init__()
        self.imagenames = os.listdir(os.path.join(dataset_folder, "Upscale"))
        self.basepath = dataset_folder
        self.device = device
        self.size = size
        self.resize = Resize(size)

    def __len__(self):
        return len(self.imagenames)

    def __getitem__(self, idx):
        x = read_image(
            os.path.join(self.basepath, "Downscale", self.imagenames[idx]),
            ImageReadMode.GRAY
        ).to(self.device)
        y = read_image(
            os.path.join(self.basepath, "Upscale", self.imagenames[idx]),
            ImageReadMode.GRAY).to(self.device)
        #               want only h and w
        if self.size == tuple(y.size()[1:]):
            return x/255, y/255
        return x/255, self.resize.forward(y)/255
