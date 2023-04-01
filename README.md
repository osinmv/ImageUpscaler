# My attempt in Upscaling with neural networks
SRCNN - stole the model
Shrinked it in size and made it grayscale, since I didn't have enough GPU memory.

The fact that I used MSE for Loss is having a negative impact on the model too.
MSE brings a lot of blur. Should experiment with other losses.
Trained for 10 epocs.

Here is one example (downscaled, upscaled, original)

![Downscaled](Assets/downscaled.jpg)
![Upscaled](Assets/upscaled.jpg)
![Original](Assets/original.jpg)