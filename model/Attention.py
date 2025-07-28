import torch
import torch.nn as nn

class ConvMFA(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class MultiFrameAttention(nn.Module):
    """(Attention layer) use convolution instead of FC layer"""
    def __init__(self, in_channels, out_channels, sub_batch_size):
        super().__init__()
        self.Wq = ConvMFA(in_channels, out_channels)#1
        self.Wk = ConvMFA(in_channels, out_channels)#2
        self.Wv = ConvMFA(in_channels, out_channels)#3
   
        self.softmax = nn.Softmax(dim=2)
        self.sub_batch_size = sub_batch_size

    def forward(self, input):
        temp = input.reshape(
            int(input.shape[0] / self.sub_batch_size),
            self.sub_batch_size,
            input.shape[1],
            input.shape[2],
            input.shape[3],
        )
        V, B, C, H, W = (
            temp.shape[0],
            temp.shape[1],
            temp.shape[2],
            temp.shape[3],
            temp.shape[4],
        )
        query = self.Wq(input).reshape(V, B, -1)
        key = self.Wk(input).reshape(V, B, -1)
        value = self.Wv(input).reshape(V, B, -1) #based traonsfromr 
        attention_weight = torch.matmul(query, key.permute(0, 2, 1)).reshape(V, B, -1)
        attention_weight = self.softmax(attention_weight)
        weighted_value = torch.matmul(attention_weight, value).reshape(V * B, C, H, W)
        return weighted_value + input, attention_weight
    
    def aggregate(self, weighted_value_plus_input, attention_weight, frame_index=1):
        """
        Aggregate the multi-frame feature maps into a single feature map
        for the reference frame (default frame_index=1).
        
        Args:
            weighted_value_plus_input (Tensor):
                [V*B, C, H, W] tensor (e.g. output of forward())
            attention_weight (Tensor):
                [V, B, B] attention matrix
            frame_index (int):
                which frame to aggregate around (0, 1, or 2)

        Returns:
            aggregated_feature (Tensor):
                [V, C, H, W]
        """
        V = attention_weight.shape[0]
        B = attention_weight.shape[1]

        # Reshape weighted_value_plus_input to [V, B, C, H, W]
        C = weighted_value_plus_input.shape[1]
        H = weighted_value_plus_input.shape[2]
        W = weighted_value_plus_input.shape[3]

        temp = weighted_value_plus_input.reshape(V, B, C, H, W)

        # Get the frame weights for the reference frame
        frame_weights = attention_weight[:, :, frame_index]   # [V, B]

        # Expand dims for broadcasting
        frame_weights = frame_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [V, B, 1, 1, 1]

        # Weighted sum across frames
        weighted_sum = (temp * frame_weights).sum(dim=1)      # [V, C, H, W]

        return weighted_sum