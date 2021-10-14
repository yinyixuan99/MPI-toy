from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io


class ToyDataset(Dataset):
    def __init__(self, device):
        super().__init__()
        self.image = torch.zeros(175, 3, 108, 192, device=device)
        self.corrd = torch.zeros(175, 2, 108, 192, device=device)
        intr = torch.tensor(np.load('THU/K/000.npy'), dtype=torch.float32)
        extr0 = torch.tensor(np.load('extrinsics.npy'), dtype=torch.float32)
        transform = (intr[:2] @ extr0[:3]).to(device)
        u, v = torch.meshgrid(torch.arange(108.), torch.arange(192.))
        mesh = torch.stack((u, v, torch.ones_like(u)), 2).unsqueeze(3)
        mesh = (intr.inverse() @ mesh).squeeze().to(device)
        corrdCam = torch.ones(108, 192, 4, device=device)
        for i in range(175):
            self.image[i] = io.read_image(f'THU/image/{i:03}.png')[:3]
            depth = np.load(f'THU/depth/{i:03}.npy')[:, :, 3]
            depth = torch.tensor(depth, dtype=torch.float32, device=device)
            extr = np.load(f'THU/pose/{i:03}.npy')
            extr = torch.tensor(extr, dtype=torch.float32, device=device)
            corrdCam[:, :, :3] = depth.unsqueeze(2) * mesh
            product = transform @ extr.inverse() @ corrdCam.unsqueeze(3)
            self.corrd[i] = product.squeeze().permute(2, 0, 1)

    def __len__(self):
        return 175

    def __getitem__(self, i):
        return self.image[i], self.corrd[i]


class ResBlock(torch.nn.Module):
    def __init__(self, inPlanes, outPlanes):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv2d(inPlanes, outPlanes, 3, padding=1),
            torch.nn.BatchNorm2d(outPlanes),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(outPlanes, outPlanes, 3, padding=1))
        self.shortcut = torch.nn.Conv2d(inPlanes, outPlanes, 1)
        self.activation = torch.nn.ReLU(True)

    def forward(self, x):
        return self.activation(self.sequential(x) + self.shortcut(x))


parser = ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch', type=int, default=7)
parser.add_argument('--epoch', type=int, default=50)
args = parser.parse_args()
device = torch.device('cuda', args.gpu)

# dataset
mpi = torch.load('mpi.pt').unsqueeze(0).repeat(args.batch, 1, 1, 1).to(device)
dataset = ToyDataset(device)
loader = DataLoader(dataset, args.batch, True)
renderer = torch.nn.Sequential(
    ResBlock(5, 64),
    ResBlock(64, 64),
    ResBlock(64, 64),
    ResBlock(64, 64),
    ResBlock(64, 3)
).to(device)

# train renderer
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(renderer.parameters())
for epoch in range(args.epoch):
    print(f'epoch = {epoch}', end='\t')
    totalLoss = 0
    for image, corrds in loader:
        input = torch.cat((mpi, corrds), 1)
        optimizer.zero_grad()
        loss = criterion(image, renderer(input))
        totalLoss += loss
        loss.backward()
        optimizer.step()
    print(f'loss = {totalLoss / 175 * args.batch}')

# test renderer
renderer.eval().requires_grad_(False)
image = torch.zeros(175, 3, 108, 192, device=device)
for i, corrds in enumerate(loader):
    input = torch.cat((mpi, corrds[1]), 1)
    image[args.batch * i:args.batch * (i + 1)] = renderer(input)
image[image < 0] = 0
image[image > 255] = 255
image = image.byte().detach().cpu()
for i in range(175):
    io.write_png(image[i], f'result/{i:03}.png')
