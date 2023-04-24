# My attempt in Upscaling with neural networks
I took the SRCNN model and reduced its size while also converting it to grayscale due to limited GPU memory. However, I have noticed that using Mean Squared Error (MSE) as the loss function has negatively affected the model's performance, resulting in excessive blur. I should try experimenting with other types of losses. I trained the model for 10 epochs.

Update: I discovered that someone used BSELoss, which allowed me to extract edges and merge them with the scaled image for slightly better results. However, the resulting image is now very dark.

Here is one example (downscaled, upscaled with nn, just scaled and original)

![Downscaled](Assets/downscaled.jpg)
![Upscaled](Assets/upscaled.jpg)
![Upscaled](Assets/scaled.jpg)
![Original](Assets/original.jpg)

Important:
Pixel diff score between stretched and original image (74.2129)
Pixel diff score between cnn upscaled and original image (70.3604)
(My upscaling performs worse than scaling)