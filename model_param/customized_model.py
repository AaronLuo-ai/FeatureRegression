import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CustomResNet34Encoder(nn.Module):
    def __init__(self, pretrained_encoder):
        super(CustomResNet34Encoder, self).__init__()

        # Copy layers from the pretrained encoder
        self.conv1 = pretrained_encoder.conv1
        self.bn1 = pretrained_encoder.bn1
        self.relu = pretrained_encoder.relu
        self.maxpool = pretrained_encoder.maxpool

        self.layer1 = pretrained_encoder.layer1
        self.layer2 = pretrained_encoder.layer2
        self.layer3 = pretrained_encoder.layer3
        self.layer4 = pretrained_encoder.layer4

        # Optional: Global Average Pooling if you need a vector output
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)  # Shape: [batch_size, 512, 1, 1]
        x = x.view(x.size(0), -1)    # Flatten to [batch_size, 512]

        return x


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        return self.linear(x)