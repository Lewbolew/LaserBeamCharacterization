import torch
import torch.nn as nn
import torch.nn.functional as F

from SegmentationModel import EESPNet_Seg
from shape_net import ShapeUNet
BN_EPS = 1e-4


class SH_UNet(nn.Module):

    def __init__(self, path_to_shape_net_weights='', n_classes=2):
        super(SH_UNet, self).__init__()

        self.unet = EESPNet_Seg(classes=2, s=2)
        self.shapeUNet = ShapeUNet((2, 1024, 1024))
        self.softmax = nn.Softmax(dim=1)
        if path_to_shape_net_weights:
            self.shapeUNet.load_state_dict(torch.load(path_to_shape_net_weights))

    def forward(self, x, only_encode=False):
        if only_encode:
            _, encoded_mask = self.shapeUNet(x)
            return encoded_mask

        if self.training:
            unet_prediction, unet_prediction_1 = self.unet(x)
        else:
            unet_prediction = self.unet(x)

        softmax_unet_prediction = self.softmax(unet_prediction)#.detach()
        shape_net_final_prediction, shape_net_encoded_prediction = self.shapeUNet(softmax_unet_prediction)

        if self.training:
            return unet_prediction, unet_prediction_1, shape_net_encoded_prediction, shape_net_final_prediction
        else:
            return unet_prediction, shape_net_encoded_prediction, shape_net_final_prediction