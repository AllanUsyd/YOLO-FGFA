# Imports
import torch
import torch.nn as nn
import torch.nn.functional as nnf

# Basic Flow estimation from feature maps

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, mode="bilinear"):
        super().__init__()

        self.mode = mode

    def forward(self, src, flow):
        device = src.device

        # create sampling grid
        size = (src.shape[2], src.shape[3])
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).to(device)

        # new locations
        new_locs = grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class DoubleConv_flow(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Flow_estimation(nn.Module):
    """Estimating flow (tgt : index=0, src : others)"""

    def __init__(self, n_channels, sub_batch):
        super().__init__()
        num = n_channels
        self.inc = DoubleConv_flow(n_channels * 2, num * 2)
        self.inc2 = DoubleConv_flow(num * 2, num * 2)
        self.inc3 = DoubleConv_flow(num * 2, num * 2)
        self.out = OutConv(num * 2, 2)
        self.warp = SpatialTransformer(mode="bilinear")
        self.sub_batch_size = sub_batch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, img):
        output = []
        flow_out = []
        img = img.reshape(
            int(img.shape[0] / self.sub_batch_size),
            self.sub_batch_size,
            img.shape[1],
            img.shape[2],
            img.shape[3],
        ).squeeze()
        if len(img.shape) != 5:
            img = img.unsqueeze(0)
        for index in range(img.shape[0]):
            tgt = img[index][0].repeat(img[index].shape[0] - 1, 1, 1, 1)
            src = img[index][1:]
            temp = torch.cat([tgt, src], dim=1)
            flow = self.out(self.inc3(self.inc2(self.inc(temp))))
            src_warp = self.warp(src, flow)
            flow_out.append(
                torch.cat(
                    [
                        torch.zeros_like(flow[0]).unsqueeze(0).to(flow.device),
                        flow,
                    ]
                )
            )
            output.append(torch.cat([img[index][0].unsqueeze(0), src_warp]))
        output = torch.cat(output)
        flow_out = torch.cat(flow_out)
        return flow_out, output