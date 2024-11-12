import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Acmf import ACMF
from .lle import LowLightEstimateNet
from .feature_transfer_nets import FeatureProjectionMLP

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch8_224.dino', pretrained=True, freeze=True):
        """
        model_name: timm model name
        pretrained: load timm pretrained model
        freeze: freeze model parameters
        
        Output: feature maps with size (1, 785, 768)
        """
        super(FeatureExtractor, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(0)  # Remove the classifier layer

        # Freeze parameters if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x): 
        
        return self.model.forward_features(x)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


# Feature Enhancement Module
class AdaptiveCrossModal(nn.Module):
    def __init__(self, feature_dim=768, num_tokens=785, pooling_type='avg', activation='Prelu'):
        super(AdaptiveCrossModal, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_tokens = num_tokens

        # Pooling layer - can switch between average and max pooling
        if pooling_type == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling_type == 'gem':
            self.pooling = GeM()
        else:
            raise ValueError("Pooling type should be either 'avg' or 'max'")

        # Activation function
        if activation == 'Prelu':
            self.activation = nn.PReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Mish':
            self.activation = nn.Mish()
        else:
            raise ValueError("Activation should be 'Prelu', 'sigmoid', or 'Mish'")

    def forward(self, feature_tensor, illumination_vector):
        # Ensure illumination_vector is reshaped for broadcasting
        illumination_vector = illumination_vector.unsqueeze(1)  # Shape: (batch_size, 1, 768)
        
        # Element-wise multiplication
        enhanced_features = feature_tensor * illumination_vector  # Shape: (batch_size, 785, 768)
        
        # Apply pooling along the token dimension
        enhanced_features = enhanced_features.permute(0, 2, 1)  # Shape: (batch_size, 768, 785)
        pooled_features = self.pooling(enhanced_features)  # Shape: (batch_size, 768, 1)
        pooled_features = pooled_features.squeeze(-1)  # Shape: (batch_size, 768)
        
        # Apply activation function
        output = self.activation(pooled_features)  # Shape: (batch_size, 768)
        
        return output



# Full Model: LowLightEnhancementNet
class MultiModalNet(nn.Module):
    def __init__(self, feature_dim=768):
        """
        Returns: well_lit_features, enhanced_features, predicted_features, predicted_upsampled_features, well_lit_upsampled_features
        """
        super(MultiModalNet, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.illumination_net = LowLightEstimateNet(feature_dim=feature_dim)
        self.feature_enhancement = AdaptiveCrossModal(feature_dim)
        self.feature_prediction = FeatureProjectionMLP(in_features=feature_dim, out_features=feature_dim)

    def forward(self, low_light_image, well_lit_image):
        # Step 1: Extract features from both low-light and well-lit images
        low_light_features = self.feature_extractor(low_light_image)
        well_lit_features = self.feature_extractor(well_lit_image)

        # Step 2: Estimate illumination from low-light features
        illumination_vector = self.illumination_net(low_light_image)

        # Step 3: Enhance low-light features using the illumination vector
        enhanced_features = self.feature_enhancement(low_light_features, illumination_vector)

        # Step 4: Predict the enhanced features using MLP
        predicted_features = self.feature_prediction(enhanced_features)

        # Step 5: Upsample predicted feature and well-lit features to 224x224
        predicted_upsampled_features = F.interpolate(predicted_features, size=(224, 224), mode='bilinear')
        well_lit_upsampled_features = F.interpolate(well_lit_features, size=(224, 224), mode='bilinear')  

        return well_lit_features, enhanced_features, predicted_features, predicted_upsampled_features, well_lit_upsampled_features
    


    def inference(self, low_light_image, well_lit_image):
        
        """
        When evaluating the model, we don't need to update the illumination vector. or enhance features.
        """
        # Step 1: Extract features from both low-light and well-lit images
        low_light_features = self.feature_extractor(low_light_image)
        well_lit_features = self.feature_extractor(well_lit_image)

        # Step 4: Predict the enhanced features using MLP
        predicted_features = self.feature_prediction(low_light_features)

        # Step 5: Upsample predicted feature and well-lit features to 224x224
        predicted_upsampled_features = F.interpolate(predicted_features, size=(224, 224), mode='bilinear')
        well_lit_upsampled_features = F.interpolate(well_lit_features, size=(224, 224), mode='bilinear')  
        
        return predicted_upsampled_features, well_lit_upsampled_features

# Example usage
if __name__ == "__main__":
    # Initialize the model
    from torch.profiler import profile, record_function, ProfilerActivity
    low_light_image = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 resolution
    well_lit_image = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 resolution
    # Enable GPU if available
    illumination_net = MultiModalNet()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")
    illumination_net = illumination_net.to(device)
    low_light_image = low_light_image.to(device)
    well_lit_image = well_lit_image.to(device)
    # Define a dummy optimizer
    optimizer = torch.optim.Adam(illumination_net.parameters(), lr=0.001)

    # Start profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True) as prof:
        # Record each part of the model's forward pass
        with record_function("model_forward"):
            illumination_vector = illumination_net(low_light_image,well_lit_image)
            # print(illumination_vector.shape)
        # Backward pass to profile gradients
        with record_function("model_backward"):
            optimizer.zero_grad()
            loss = illumination_vector[0].mean() + illumination_vector[1].sum()
             # Dummy loss
            loss.backward()
            optimizer.step()

    # Print profiling results
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    
    