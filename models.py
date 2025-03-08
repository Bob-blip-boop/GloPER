import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class UNetColorCentroidLocal2d(nn.Module):
    def __init__(self, num_classes, local_size = 4, image_size = (1024,1024), input_channels=3, **kwargs):
        super(UNetColorCentroidLocal, self).__init__()
        
        nb_filter = [32, 64, 128, 256, 512]

        self.num_classes = num_classes
        self.local_size = local_size
        self.h_regions = image_size[0] // self.local_size  # Height divided by region size
        self.w_regions = image_size[1] // self.local_size 

        self.h_regions_1 = image_size[0] // (self.local_size * 2)  # Height divided by region size
        self.w_regions_1 = image_size[1] // (self.local_size * 2)


        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder layers
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        # Decoder layers with skip connections
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        # Final segmentation output
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        # RGB centroid prediction branch
        self.centroid_pool = nn.AdaptiveAvgPool2d((self.h_regions, self.w_regions))  # Local pooling
        self.centroid_fc_0 = nn.Conv2d(nb_filter[0], 2 * 3, kernel_size=1)  # One set of RGB centroids per local area
        self.centroid_pool_1 = nn.AdaptiveAvgPool2d((self.h_regions_1, self.w_regions_1))  # Local pooling
        self.centroid_fc_1 = nn.Conv2d(nb_filter[1], 2 * 3, kernel_size=1)  # One set of RGB centroids per local area
        

    def forward(self, x):
        # print(x.shape)
        # Encoding path
        x0_0 = self.conv0_0(x)                      # torch.Size([1, 32, 512, 512]) Low-level details (edges, corners)
        x1_0 = self.conv1_0(self.pool(x0_0))        # torch.Size([1, 64, 256, 256])
        x2_0 = self.conv2_0(self.pool(x1_0))        # torch.Size([1, 128, 128, 128])
        x3_0 = self.conv3_0(self.pool(x2_0))        # torch.Size([1, 256, 64, 64])
        x4_0 = self.conv4_0(self.pool(x3_0))        # torch.Size([1, 512, 32, 32]) High-level features (regions, objects)

        # Decoding path with skip connections
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        # Segmentation output
        segmentation_output = self.final(x0_4)

        # RGB centroid prediction
        pooled_features_x0 = self.centroid_pool(x0_0)  # Global average pool the bottleneck features
        centroid_logits_x0 = self.centroid_fc_0(pooled_features_x0)  # Shape: [batch_size, num_classes * 3, 1, 1]
        centroid_output_x0 = torch.sigmoid(centroid_logits_x0) 
        centroid_output_x0 = centroid_output_x0.view(-1, self.num_classes, 3, self.h_regions, self.w_regions)  # Reshape to [batch_size, num_classes, 3]

        pooled_features_x1 = self.centroid_pool_1(x1_0)  # Global average pool the bottleneck features
        centroid_logits_x1 = self.centroid_fc_1(pooled_features_x1)  # Shape: [batch_size, num_classes * 3, 1, 1]
        centroid_output_x1 = torch.sigmoid(centroid_logits_x1) 
        centroid_output_x1 = centroid_output_x1.view(-1, self.num_classes, 3, self.h_regions_1, self.w_regions_1)  # Reshape to [batch_size, num_classes, 3]

        # return weight_output.squeeze(0), segmentation_output, centroid_output_x0, centroid_output_x4
        return segmentation_output, centroid_output_x0, centroid_output_x1


