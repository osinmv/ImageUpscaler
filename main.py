import torch
from dataset import UpscalerDataset
mps_device = torch.device("mps")
model = torch.nn.Sequential(
    torch.nn.UpsamplingBilinear2d(size=(4800, 6400)),
    torch.nn.Conv2d(1, 1, kernel_size=3,stride=2),
    torch.nn.Conv2d(1, 1, kernel_size=3, stride=2,padding=1),
)

model.to(mps_device)
data = UpscalerDataset("Data", mps_device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.L1Loss().to(mps_device)
dataset = torch.utils.data.DataLoader(data, batch_size=16)
for ep in range(0, 30):
    for i, data in enumerate(dataset):
        xs, ys = data
        optimizer.zero_grad()
        out = model(xs)
        loss = loss_fn(out, ys)
        loss.backward()
        optimizer.step()
        print(ep, loss)
torch.save(model, "weight")
