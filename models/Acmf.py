
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity


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
            kernel_size=3,
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
            kernel_size=3,
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
        out = out_upsample.permute(0, 2, 3, 1) # (batch, H, W, 768)
        return out


if __name__ == "__main__":
    # Define a dummy input for testing
    feature_dim = 768
    illumination_net = ACMF(in_channels_evt=feature_dim, in_channels_img=feature_dim)

    # Example low-light image tensor for input
    low_light_image = torch.randn(2, 768 ,224,224)  # Batch size 1, 3 color channels, 224x224 resolution

    # Enable GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    illumination_net = illumination_net.to(device)
    low_light_image = low_light_image.to(device)

    # Define a dummy optimizer
    optimizer = torch.optim.Adam(illumination_net.parameters(), lr=0.001)

    # Start profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True) as prof:
        # Record each part of the model's forward pass
        with record_function("model_forward"):
            illumination_vector = illumination_net(low_light_image, low_light_image)

        # Backward pass to profile gradients
        with record_function("model_backward"):
            optimizer.zero_grad()
            loss = illumination_vector.sum()  # Dummy loss
            loss.backward()
            optimizer.step()

    # Print profiling results
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))