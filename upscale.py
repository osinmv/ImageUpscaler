import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from torchvision.transforms import Resize

mps_device = torch.device("mps")
model = torch.load("weight")
image_tensor = (read_image(
    "Data/Downscale/Image00001.jpg", ImageReadMode.GRAY)/255).to(mps_device)
img = torch.reshape(image_tensor, (1, 1, 450, 600))
edges = model.forward(img)
img = Resize((1200,1600)).forward(img)
result = edges * img*2+0.1
save_image(result, "image.jpeg", "jpeg")
