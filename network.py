from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch.nn as nn

class Network(nn.Module):
    """
    Neural network class using MobileNetV3 Large as the backbone.
    """

    def __init__(self):
        super(Network, self).__init__()
        num_classes = 22    # Define the number of output classes

        # Load the pre-trained MobileNetV3 Large model with ImageNet weights
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        # Freeze the feature extractor layers to prevent them from training
        for param in self.model.features.parameters():
            param.requires_grad = False

         # Replace the classifier with custom layers to adapt to our specific task
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[0].in_features, 512),   # First fully connected layer
            nn.ReLU(),                                              # Activation function
            nn.Dropout(0.2),                                        # Dropout layer for regularization
            nn.Linear(512, num_classes)                             # Output layer with 'num_classes' outputs
        )

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor representing a batch of images.
        Returns:
            torch.Tensor: Output logits from the network.
        """
        
        return self.model(x)

""" 
The following are the baseline networks we did not end up using (see report for analysis!)
In order:
- DenseNet
- EfficientNet
- AlexNet
- SqueezeNet
- ViT Transformer
- ConvNeXt
- ResNet50
"""
# DenseNet
"""
from torchvision.models import densenet121, DenseNet121_Weights
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        num_classes = 22

        self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

"""

# EfficientNet
"""
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        num_classes = 22

        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

"""

# AlexNet
"""
from torchvision.models import alexnet, AlexNet_Weights
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        num_classes = 22 

        self.model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)
"""

# SqueezeNet
"""
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        num_classes = 22  

        self.model = squeezenet1_0(weights=SqueezeNet1_0_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)
"""

# ViT Transformer
"""
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        num_classes = 22 

        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        self.model.heads = nn.Sequential(
            nn.Linear(self.model.heads.head.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

"""

# ConvNeXt
""" 
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        num_classes = 22 

        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

"""

# ResNet50
""" 
()
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Network(nn.Module):
    def __init__(self, num_classes=22, freeze_base=True):
        super(Network, self).__init__()

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

"""