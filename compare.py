import numpy as np
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize


def get_pixel_wise_similaroty(orig, upscaled):
    diff = (orig - upscaled) ** 2
    mean = diff.mean()
    return 10. * np.log(1. / mean)


downscaled = read_image("Data/Downscale/Image00001.jpg",
                        ImageReadMode.GRAY)/255
cnn_upscaled = read_image("./Assets/upscaled.jpg", ImageReadMode.GRAY)/255
upscaled = Resize((1200, 1600)).forward(downscaled)
original = read_image("Data/Upscale/Image00001.jpg", ImageReadMode.GRAY)/255


print("Pixel diff score between stretched and original",
      get_pixel_wise_similaroty(original, upscaled))
print("Pixel diff score between cnn upscaled and original",
      get_pixel_wise_similaroty(original, cnn_upscaled))
