import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(x.mean((2, 3), keepdim=True))))
        max_out = self.fc2(self.relu(self.fc1(x.amax((2, 3), keepdim=True))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class ACMF(nn.Module):
    def __init__(self, in_channels_img, in_channels_evt, reduction=16):
        super(ACMF, self).__init__()
        # Initial convolution to merge image and event features
        self.conv1 = nn.Conv2d(
            in_channels_img + in_channels_evt,
            in_channels_img,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # Average pooling to downscale features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Channel and spatial attention modules
        self.ca = ChannelAttention(in_channels_img, reduction=reduction)
        self.sa = SpatialAttention()

        # Final convolution to refine enhanced features
        self.conv2 = nn.Conv2d(
            in_channels_img,
            in_channels_img,
            kernel_size=1,
        )

        # self.conv3 = nn.Conv2d(in_channels_evt)

    def forward(self, F_img_t, F_evt_t):
        # Concatenate image and event features along the channel dimension
        cat_fea = torch.cat(
            (F_img_t, F_evt_t), dim=1
        )  # Shape: (batch, in_channels_img + in_channels_evt, H, W)

        # Apply the first convolution
        cat_fea = self.conv1(cat_fea)
        # print(f"after conv1 {cat_fea.shape}")
        # Apply average pooling
        cat_fea = self.avg_pool(cat_fea)  # Shape: bx768x1x1
        # print(cat_fea.shape)
        # Either split tensor or just multiply Element-wise
        F_evt_t_mul = F_evt_t * cat_fea
        F_img_t_mul = F_img_t * cat_fea
        # Apply channel attention and spatial attention sequentially
        ca_out = self.ca(F_evt_t_mul)  # Channel attention output
        sa_out = self.sa(ca_out)  # Spatial attention output

        # Multiply the attention maps with the input feature map
        x = F_img_t_mul + sa_out

        # Apply the second convolution
        x = self.conv2(x)
        evt = self.conv2(F_img_t_mul)
        # Upsample to match the original spatial dimensions of `F_img_t`
        # x = F.interpolate(
        #     x, size=F_img_t.shape[2:], mode="bilinear", align_corners=False
        # )

        # Element-wise sum with the original image features
        out = evt + x
        out_upsample = torch.nn.functional.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        out = out_upsample.reshape(out_upsample.shape[1], -1).T
        return out


if __name__ == "__main__":
    # Define a dummy input for testing
    batch_size = 1
    in_channels_img = 768
    in_channels_evt = 768

    F_img_t = torch.randn(batch_size, 768,28, 28)
    F_evt_t = torch.randn(batch_size, 768, 28, 28)

    # Initialize the ACMF module
    acmf = ACMF(in_channels_img, in_channels_evt)
    # Test the forward pass
    output = acmf(F_img_t, F_evt_t)
    print(output.shape)
