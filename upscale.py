import torch
from torchvision.io import read_image
from torchvision.utils import save_image


mps_device = torch.device("mps")
model = torch.load("weight")
image_tensor = (read_image(
    "Data/Downscale/Image00001.jpg")/255).to(mps_device)
img = torch.reshape(image_tensor, (1, 3, 450, 600))
save_image(model.forward(img), "image.jpeg", "jpeg")
