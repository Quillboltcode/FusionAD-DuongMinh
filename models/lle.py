import torch
import torch.nn as nn
import torch.nn.functional as F

# Debug tools
from torch.profiler import profile, record_function, ProfilerActivity

# using mobilenet v2 inverted residual blocks
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride=1):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expansion_factor
        
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Pointwise expansion
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        
        # Pointwise projection
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = F.relu6(self.expand_bn(self.expand_conv(x)))
        out = F.relu6(self.depthwise_bn(self.depthwise_conv(out)))
        out = self.project_bn(self.project_conv(out))
        
        if self.use_residual:
            return x + out
        else:
            return out

class LowLightEstimateNet(nn.Module):
    def __init__(self, feature_dim):
        super(LowLightEstimateNet, self).__init__()
        
        # Input layer
        self.initial_conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # Reduced from 32 to 16
        
        # Series of inverted residual blocks with reduced channels
        self.block1 = InvertedResidualBlock(16, 32, expansion_factor=4, stride=2)  # 16 -> 32
        self.block2 = InvertedResidualBlock(32, 64, expansion_factor=4, stride=2)  # 32 -> 64
        self.block3 = InvertedResidualBlock(64, 128, expansion_factor=4, stride=2) # 64 -> 128
        self.block4 = InvertedResidualBlock(128, 256, expansion_factor=4, stride=2) # 128 -> 256
        


        # Fully connected layers to produce illumination vector
        self.fc1 = nn.Linear(256 * 7 * 7, 512)  # Reduced from 512*7*7 and 1024
        self.fc2 = nn.Linear(512, feature_dim)
        
    def forward(self, low_light_img):
        # Initial convolution layer
        x = F.relu6(self.initial_conv(low_light_img))
        
        # Pass through inverted residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Flatten and pass through fully connected layers to generate illumination vector
        x = x.view(x.size(0), -1)
        x = F.relu6(self.fc1(x))
        illumination_vector = torch.sigmoid(self.fc2(x))  # Sigmoid to normalize values between 0 and 1
        
        return illumination_vector





if __name__ == "__main__":
    # Instantiate the model
    feature_dim = 768
    illumination_net = LowLightEstimateNet(feature_dim)

    # Example low-light image tensor for input
    low_light_image = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 resolution

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
            illumination_vector = illumination_net(low_light_image)
            print(illumination_vector.shape)
        # Backward pass to profile gradients
        with record_function("model_backward"):
            optimizer.zero_grad()
            loss = illumination_vector.sum()  # Dummy loss
            loss.backward()
            optimizer.step()

    # Print profiling results
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))