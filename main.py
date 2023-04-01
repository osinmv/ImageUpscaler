import torch
from dataset import UpscalerDataset

mps_device = torch.device("mps")
model = torch.nn.Sequential(
    torch.nn.UpsamplingBilinear2d(size=(1200,1600)),
        torch.nn.Conv2d(3,16,kernel_size=9,padding=4),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16,8,kernel_size=1,padding=0),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8,3,kernel_size=5,padding=2),
)

model.to(mps_device)
data = UpscalerDataset("Data", mps_device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss().to(mps_device)
dataset = torch.utils.data.DataLoader(data, batch_size=16)
for ep in range(0, 10):
    for i, data in enumerate(dataset):
        xs, ys = data
        optimizer.zero_grad()
        out = model(xs)
        loss = loss_fn(out, ys)
        loss.backward()
        optimizer.step()
        print(ep, loss)
torch.save(model, "weight")
