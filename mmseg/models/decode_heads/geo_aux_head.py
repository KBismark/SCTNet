import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.cnn.utils.weight_init import constant_init, kaiming_init

from ..builder import HEADS
from .decode_head import BaseDecodeHead

from scipy.ndimage import distance_transform_edt
import numpy as np
import cv2


@HEADS.register_module()
class GeoAuxHead(BaseDecodeHead):
    """
    Predicts a per-class signed distance transform and a
    class boundary map from a backbone feature, supervised
    against ground truth. Contributes to the loss
    only during training; 
    """

    def __init__(self,
                 prior_loss_weight=0.15,
                 boundary_loss_weight=0.10,
                 boundary_dilate_size=2,
                 **kwargs):
        # num_classes is required by BaseDecodeHead's __init__ (builds
        # self.conv_seg with that many output channels) -- we don't use
        # conv_seg/cls_seg at all (this head predicts SDT/boundary, not a
        # class mask), but BaseDecodeHead unconditionally constructs it,
        # so we let it build normally and simply never call cls_seg().
        super(GeoAuxHead, self).__init__(**kwargs)
        self.prior_loss_weight = prior_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.boundary_dilate_size = boundary_dilate_size

        self.bn1 = nn.SyncBatchNorm(self.in_channels)
        self.relu = nn.ReLU()
        self.prior_conv1 = Conv2d(self.in_channels, self.channels, kernel_size=3, padding=1)
        self.prior_bn2 = nn.SyncBatchNorm(self.channels)
        self.prior_out = Conv2d(self.channels, self.num_classes, kernel_size=1)

        self.boundary_conv1 = Conv2d(self.in_channels, self.channels, kernel_size=3, padding=1)
        self.boundary_bn2 = nn.SyncBatchNorm(self.channels)
        self.boundary_out = Conv2d(self.channels, 1, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
                constant_init(m.weight, val=1.0)
                constant_init(m.bias, val=0)
            elif isinstance(m, nn.Conv2d):
                kaiming_init(m.weight)
                if m.bias is not None:
                    constant_init(m.bias, val=0)

    def forward(self, inputs):
        """Returns (prior, boundary), NOT a segmentation logit -- this
        head never calls self.cls_seg(), unlike SCTHead/AU_SCTHead."""
        x = self._transform_inputs(inputs)
        x = self.relu(self.bn1(x))

        prior = self.prior_conv1(x)
        prior = self.relu(self.prior_bn2(prior))
        prior = torch.tanh(self.prior_out(prior))

        boundary = self.boundary_conv1(x)
        boundary = self.relu(self.boundary_bn2(boundary))
        boundary = torch.sigmoid(self.boundary_out(boundary))

        return prior, boundary

    @staticmethod
    def _make_geo_targets(seg_label, num_classes, ignore_index, dilate_size):
        """
        seg_label: (N, 1, H, W) or (N, H, W) long tensor of class ids.
        Returns (distance, boundary_gt) matching the network's own
        low-resolution feature map is NOT required here -- both are
        computed at seg_label's native resolution then interpolated to
        match prediction resolution in forward_train(), so this function
        doesn't need to know the backbone's stride.
        """

        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        label_np = seg_label.detach().cpu().numpy()
        n, h, w = label_np.shape

        distance = np.zeros((n, num_classes, h, w), dtype=np.float32)
        boundary = np.zeros((n, 1, h, w), dtype=np.float32)

        for b in range(n):
            lm = label_np[b]
            valid = lm != ignore_index

            for c in range(num_classes):
                binary = ((lm == c) & valid).astype(np.uint8)
                if binary.sum() == 0 or binary.sum() == valid.sum():
                    distance[b, c] = -1.0 if binary.sum() == 0 else 1.0
                    continue
                dist_in = distance_transform_edt(binary)
                dist_out = distance_transform_edt(1 - binary)
                sdf = dist_in - dist_out
                max_val = max(abs(sdf.max()), abs(sdf.min())) + 1e-6
                distance[b, c] = (sdf / max_val).astype(np.float32)

            lm_i = lm.astype(np.int32).copy()
            lm_i[lm_i == ignore_index] = -1
            bmap = np.zeros_like(lm_i, dtype=np.uint8)
            bmap[:-1, :] |= (lm_i[:-1, :] != lm_i[1:, :])
            bmap[1:, :] |= (lm_i[:-1, :] != lm_i[1:, :])
            bmap[:, :-1] |= (lm_i[:, :-1] != lm_i[:, 1:])
            bmap[:, 1:] |= (lm_i[:, :-1] != lm_i[:, 1:])
            bmap[lm_i == -1] = 0
            if dilate_size > 1:
                kernel = np.ones((dilate_size, dilate_size), np.uint8)
                bmap = cv2.dilate(bmap, kernel, iterations=1)
            boundary[b, 0] = bmap.astype(np.float32)

        device = seg_label.device
        return (torch.from_numpy(distance).to(device),
                torch.from_numpy(boundary).to(device))

    def forward_train(self, inputs, decoder_feature, decoder_seg_logits,
                       img_metas, gt_semantic_seg, train_cfg):
        
        prior, boundary = self.forward(inputs)

        distance_gt, boundary_gt = self._make_geo_targets(
            gt_semantic_seg, self.num_classes, self.ignore_index, self.boundary_dilate_size
        )

        # match spatial resolution -- prior/boundary are at the tapped
        # feature's stride, GT was computed at seg_label's native resolution 
        if prior.shape[-2:] != distance_gt.shape[-2:]:
            distance_gt = F.interpolate(distance_gt, size=prior.shape[-2:], mode='nearest')
            boundary_gt = F.interpolate(boundary_gt, size=boundary.shape[-2:], mode='nearest')

        loss_prior = F.smooth_l1_loss(prior, distance_gt)
        loss_boundary = F.binary_cross_entropy(boundary, boundary_gt)

        losses = dict(
            loss_prior=loss_prior * self.prior_loss_weight,
            loss_boundary=loss_boundary * self.boundary_loss_weight,
        )
        return losses
    
    