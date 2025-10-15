import math
import os
import sys

import torch
import torch.nn as nn

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError as exc:  # pragma: no cover - dependency provided in training env
    raise ImportError(
        "diff_gaussian_rasterization must be installed to use GaussianLSS."
    ) from exc

try:
    from .bev_pool import bev_pool
    from .img_encoders import SwinT_tiny_Encoder
except Exception:  # pragma: no cover - fallback for standalone usage
    import bev_pool.bev_pool as bev_pool
    from img_encoders import SwinT_tiny_Encoder


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bev_settings import d_conf, down_ratio, sparse_shape, vsize_xyz, xbound, ybound, zbound
from proj_head import ProjectionHead

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx

class LSS(nn.Module):
    ### Adapted from https://github.com/nv-tlabs/lift-splat-shoot
    def __init__(self, 
                 transformedImgShape = (3, 256, 704),
                 featureShape = (256, 32, 88),
                 d_conf = d_conf,
                 out_channels = 128,
                 ):
        super(LSS, self).__init__()
        _, self.orfH, self.orfW = transformedImgShape
        self.fC, self.fH, self.fW = featureShape
        self.d_st, self.d_end, self.d_step = d_conf
        self.D = torch.arange(self.d_st, self.d_end, self.d_step, dtype = torch.float).shape[0]
        self.out_channels = out_channels
        self.frustum = self.create_frustum()
        self.depth_net = nn.Sequential(
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, self.out_channels + self.D, 1),
        )
    
    def create_frustum(self):
        ds = torch.arange(self.d_st, self.d_end, self.d_step, dtype = torch.float).view(-1, 1, 1).expand(-1, self.fH, self.fW) # (D, fH, fW)
        xs = torch.linspace(0, self.orfW - 1, self.fW).view(1, 1, self.fW).expand(self.D, self.fH, self.fW)
        ys = torch.linspace(0, self.orfH - 1, self.fH).view(1, self.fH, 1).expand(self.D, self.fH, self.fW)

        frustum = torch.stack((xs, ys, ds), -1) # img space, the frustum is a cuboid.
        return nn.Parameter(frustum, requires_grad = False)

    def get_geometry(self, 
                     cam2ego_rot,
                     cam2ego_trans,
                     cam_intrins,
                     post_cam2ego_rot,
                     post_cam2ego_trans,
                     ):
        img_rots, img_trans, cam_intrins, img_post_rots, img_post_trans = cam2ego_rot, cam2ego_trans, cam_intrins, post_cam2ego_rot, post_cam2ego_trans
        B, N, _ = img_trans.shape
        points = self.frustum - img_post_trans.view(B, N, 1, 1, 1, 3) # (B, N, D, fH, fW, 3)
        points = torch.inverse(img_post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # before : (B, N, D, fH, fW, (x, y, z), 1), image space
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], # x * z, y * z
                points[:, :, :, :, :, 2:3] # z
            ),
            5,
        )
        # after : (B, N, D, fH, fW, (x * z, y * z, z), 1), transfrom from image space to camera space, the frustum transforms from cuboid to pyramid. 

        combine = img_rots.matmul(torch.inverse(cam_intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += img_trans.view(B, N, 1, 1, 1, 3) # To ego space

        return points 

    def get_cam_feature(self, img_feats):
        B, N, C, H, W = img_feats.shape
        img_feats = img_feats.view(B * N, C, H, W) 
        img_feats = self.depth_net(img_feats) # (B * N, out_channels + D, H, W)
        depth = img_feats[:, :self.D].softmax(dim = 1) # (B * N, D, H, W)
        img_feats = depth.unsqueeze(1) * img_feats[:, self.D : self.D + self.out_channels].unsqueeze(2) # (BN, 1, D, H, W) * (BN, out_channels, 1, H, W) => (BN, out_channels, D, H, W)
        img_feats = img_feats.view(B, N, self.out_channels, self.D, H, W)
        img_feats = img_feats.permute(0, 1, 3, 4, 5, 2)
        
        return img_feats

    def forward(self, 
                cam2ego_rot,
                cam2ego_trans,
                cam_intrins,
                post_cam2ego_rot,
                post_cam2ego_trans,
                img_feats
                ):
        geometry = self.get_geometry(cam2ego_rot=cam2ego_rot, cam2ego_trans=cam2ego_trans, cam_intrins=cam_intrins, post_cam2ego_rot=post_cam2ego_rot, post_cam2ego_trans=post_cam2ego_trans)
        img_depth_feature = self.get_cam_feature(img_feats)
        return geometry, img_depth_feature

class GaussianLSS(nn.Module):
    """Gaussian Lift-Splat-Shoot module with differentiable splatting.

    This implementation adapts the reference project at
    https://github.com/HCIS-Lab/GaussianLSS to the current codebase. Each image
    pixel generates a single 3D Gaussian in the ego frame whose mean and
    covariance are obtained by marginalising a discretised depth distribution.
    The Gaussians are then rendered into the bird's-eye-view (BEV) plane using
    the differentiable Gaussian rasteriser provided by
    ``diff_gaussian_rasterization``.
    """

    def __init__(self,
                 transformedImgShape=(3, 256, 704),
                 featureShape=(256, 32, 88),
                 d_conf=d_conf,
                 out_channels=128,
                 error_tolerance=1.0,
                 opacity_filter=0.05,
                 ):
        super().__init__()
        _, self.orfH, self.orfW = transformedImgShape
        self.fC, self.fH, self.fW = featureShape
        self.d_st, self.d_end, self.d_step = d_conf
        self.D = torch.arange(self.d_st, self.d_end, self.d_step, dtype=torch.float32).shape[0]
        self.out_channels = out_channels

        self.frustum = self.create_frustum()

        self.depth_net = nn.Sequential(
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, self.out_channels + self.D + 1, 1),
        )

        bev_h = int((ybound[1] - ybound[0]) / ybound[2])
        bev_w = int((xbound[1] - xbound[0]) / xbound[2])
        self.renderer = GaussianRenderer(
            embed_dims=self.out_channels,
            bev_shape=(bev_h, bev_w),
            threshold=opacity_filter,
        )

        x_center = (xbound[0] + xbound[1]) / 2.0
        y_center = (ybound[0] + ybound[1]) / 2.0
        self.register_buffer(
            "bev_center",
            torch.tensor([x_center, y_center, 0.0], dtype=torch.float32),
            persistent=False,
        )

        scale_x = 2.0 / (xbound[1] - xbound[0])
        scale_y = 2.0 / (ybound[1] - ybound[0])
        self.register_buffer(
            "bev_scale",
            torch.tensor([scale_x, scale_y, 1.0], dtype=torch.float32),
            persistent=False,
        )

        self.error_tolerance = error_tolerance

    def create_frustum(self):
        ds = torch.arange(self.d_st, self.d_end, self.d_step, dtype=torch.float32).view(-1, 1, 1).expand(-1, self.fH, self.fW)
        xs = torch.linspace(0, self.orfW - 1, self.fW).view(1, 1, self.fW).expand(self.D, self.fH, self.fW)
        ys = torch.linspace(0, self.orfH - 1, self.fH).view(1, self.fH, 1).expand(self.D, self.fH, self.fW)

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self,
                     cam2ego_rot,
                     cam2ego_trans,
                     cam_intrins,
                     post_cam2ego_rot,
                     post_cam2ego_trans,
                     ):
        img_rots, img_trans, cam_intrins, img_post_rots, img_post_trans = cam2ego_rot, cam2ego_trans, cam_intrins, post_cam2ego_rot, post_cam2ego_trans
        B, N, _ = img_trans.shape
        points = self.frustum - img_post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(img_post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3]
            ),
            5,
        )

        combine = img_rots.matmul(torch.inverse(cam_intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += img_trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feature(self, img_feats):
        B, N, C, H, W = img_feats.shape
        img_feats = img_feats.view(B * N, C, H, W)
        img_feats = self.depth_net(img_feats)

        depth_logits = img_feats[:, :self.D]
        opacity_logit = img_feats[:, self.D:self.D + 1]
        img_feats = img_feats[:, self.D + 1:]

        depth_prob = depth_logits.softmax(dim=1)
        opacities = opacity_logit.sigmoid()

        depth_prob = depth_prob.view(B, N, self.D, H, W)
        opacities = opacities.view(B, N, 1, H, W)
        img_feats = img_feats.view(B, N, self.out_channels, H, W)

        img_depth_feature = depth_prob.unsqueeze(-1) * img_feats.unsqueeze(2)

        return img_feats, img_depth_feature, depth_prob, opacities

    def compute_gaussian(self, geometry, depth_prob):
        weights = depth_prob.unsqueeze(-1)
        means3d = (weights * geometry).sum(2)
        delta = means3d.unsqueeze(2) - geometry
        cov = (weights.unsqueeze(-1) * (delta.unsqueeze(-1) @ delta.unsqueeze(-2))).sum(2)

        scale = (self.error_tolerance ** 2) / 9.0
        cov = cov * scale

        return means3d, cov

    def forward(self,
                cam2ego_rot,
                cam2ego_trans,
                cam_intrins,
                post_cam2ego_rot,
                post_cam2ego_trans,
                img_feats
                ):
        geometry = self.get_geometry(
            cam2ego_rot=cam2ego_rot,
            cam2ego_trans=cam2ego_trans,
            cam_intrins=cam_intrins,
            post_cam2ego_rot=post_cam2ego_rot,
            post_cam2ego_trans=post_cam2ego_trans,
        )
        img_feats, img_depth_feature, depth_prob, opacities = self.get_cam_feature(img_feats)

        means3d, cov3d = self.compute_gaussian(geometry, depth_prob)

        bev_center = self.bev_center.view(1, 1, 1, 1, 3)
        bev_scale = self.bev_scale.view(1, 1, 1, 1, 3)
        means3d_norm = (means3d - bev_center) * bev_scale

        scale_outer = bev_scale.unsqueeze(-1) * bev_scale.unsqueeze(-2)
        cov3d_norm = cov3d * scale_outer

        cov3d_norm = cov3d_norm.flatten(-2, -1)
        cov3d_norm = torch.cat(
            (
                cov3d_norm[..., 0:3],
                cov3d_norm[..., 4:6],
                cov3d_norm[..., 8:9],
            ),
            dim=-1,
        )

        features = img_feats.permute(0, 1, 3, 4, 2)
        features = features.reshape(features.shape[0], -1, self.out_channels)
        means3d_norm = means3d_norm.reshape(means3d_norm.shape[0], -1, 3)
        cov3d_norm = cov3d_norm.reshape(cov3d_norm.shape[0], -1, 6)
        opacities = opacities.permute(0, 1, 3, 4, 2).reshape(opacities.shape[0], -1, 1)

        bev_features, num_gaussians = self.renderer(features, means3d_norm, cov3d_norm, opacities)

        return {
            "geometry": geometry,
            "img_depth_feature": img_depth_feature.contiguous(),
            "bev_features": bev_features,
            "opacities": opacities,
            "num_gaussians": num_gaussians,
        }


class BEVCamera:
    def __init__(self, x_range=(-1, 1), y_range=(-1, 1), image_size=(200, 200)):
        if isinstance(image_size, tuple):
            self.image_height, self.image_width = image_size
        else:
            self.image_height = self.image_width = image_size

        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

        self.FoVx = (self.x_max - self.x_min)
        self.FoVy = (self.y_max - self.y_min)

        self.camera_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        self.set_transform(self.image_height, self.image_width, self.FoVy, self.FoVx)

    def set_transform(self, h=200, w=200, h_meters=2.0, w_meters=2.0):
        sh = h / h_meters
        sw = w / w_meters
        self.world_view_transform = torch.tensor([
            [0.0, sh, 0.0, 0.0],
            [sw, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float32)

        self.full_proj_transform = torch.tensor([
            [0.0, -sh, 0.0, h / 2.0],
            [-sw, 0.0, 0.0, w / 2.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)

    def set_size(self, h, w):
        self.image_height = h
        self.image_width = w


class GaussianRenderer(nn.Module):
    def __init__(self, embed_dims, bev_shape=(200, 200), threshold=0.05):
        super().__init__()
        self.embed_dims = embed_dims
        self.threshold = threshold
        self.viewpoint_camera = BEVCamera(image_size=bev_shape)
        self.rasterizer = GaussianRasterizer()

    def forward(self, features, means3D, cov3D, opacities):
        b = features.shape[0]
        device = means3D.device

        bev_out = []
        num_gaussians = []

        mask = (opacities > self.threshold).squeeze(-1)

        self.set_render_scale(self.viewpoint_camera.image_height, self.viewpoint_camera.image_width)
        self.set_rasterizer(device)

        for batch_idx in range(b):
            valid = mask[batch_idx]
            if not torch.any(valid):
                rendered_bev = torch.zeros(
                    self.embed_dims,
                    int(self.viewpoint_camera.image_height),
                    int(self.viewpoint_camera.image_width),
                    device=device,
                    dtype=features.dtype,
                )
            else:
                rendered_bev, _ = self.rasterizer(
                    means3D=means3D[batch_idx][valid],
                    means2D=None,
                    shs=None,
                    colors_precomp=features[batch_idx][valid],
                    opacities=opacities[batch_idx][valid],
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D[batch_idx][valid],
                )
            bev_out.append(rendered_bev)
            num_gaussians.append(valid.detach().float().sum())

        bev = torch.stack(bev_out, dim=0)
        num_gaussians = torch.stack(num_gaussians).mean().cpu()

        return bev, num_gaussians

    @torch.no_grad()
    def set_rasterizer(self, device):
        tanfovx = math.tan(self.viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(self.viewpoint_camera.FoVy * 0.5)

        bg_color = torch.zeros((self.embed_dims), device=device)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.viewpoint_camera.image_height),
            image_width=int(self.viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=self.viewpoint_camera.world_view_transform.to(device),
            projmatrix=self.viewpoint_camera.full_proj_transform.to(device),
            sh_degree=0,
            campos=self.viewpoint_camera.camera_center.to(device),
            prefiltered=False,
            debug=False,
        )
        self.rasterizer.set_raster_settings(raster_settings)

    @torch.no_grad()
    def set_render_scale(self, h, w):
        self.viewpoint_camera.set_size(h, w)
        self.viewpoint_camera.set_transform(h, w, self.viewpoint_camera.FoVy, self.viewpoint_camera.FoVx)

class Cam2BEV(nn.Module):
    def __init__(self, 
                 output_indices = [1, 2, 3], 
                 featureShape = (256, 32, 88), 
                 encoder_out_channels = 256,
                 FPN_in_channels = [192, 384, 768], 
                 FPN_out_channels = 256,
                 ):
        super(Cam2BEV, self).__init__()
        self.lss = GaussianLSS()
        self.CamEncode = SwinT_tiny_Encoder(output_indices, featureShape, encoder_out_channels, FPN_in_channels, FPN_out_channels)
        dx, bx, nx = gen_dx_bx(xbound=xbound, ybound=ybound, zbound=zbound)
        self.dx = nn.Parameter(dx, requires_grad = False)
        self.bx = nn.Parameter(bx, requires_grad = False)
        self.nx = nn.Parameter(nx, requires_grad = False)
        self.proj_head = ProjectionHead(embedding_dim=self.lss.out_channels)
        self.out_channels = self.proj_head.projection_dim
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        print(f"cam bev resolution: {nx[0].item()} x {nx[1].item()} x {nx[2].item()}")

    def bev_pool(self, img_depth_feature, geometry):
        img_pc = img_depth_feature
        geom_feats = geometry
        B, N, D, H, W, C = img_pc.shape
        Nprime = B * N * D * H * W
        img_pc = img_pc.reshape(Nprime, C)

        # Align the geometry to the voxel grid
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=img_pc.device, dtype=torch.long)
                for ix in range(B)
            ]
        )   
        geom_feats = torch.cat([geom_feats, batch_ix], 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        img_pc = img_pc[kept]
        geom_feats = geom_feats[kept]

        try:
            cam_bev = bev_pool(img_pc, geom_feats, B, self.nx[2], self.nx[0], self.nx[1]) # (B, out_channels, nx[2], nx[0], nx[1])
        except:
            cam_bev = torch.zeros(B, C, self.nx[2].item(), self.nx[0].item(), self.nx[1].item(), device = img_pc.device, dtype = img_pc.dtype)

        cam_bev = torch.cat(cam_bev.unbind(dim = 2), 1)

        return cam_bev
    
    def ref_bev_pool(self, img_depth_feature, geometry):
        with torch.no_grad():
            img_pc = img_depth_feature
            geom_feats = geometry
            B, N, D, H, W, C = img_pc.shape
            Nprime = B * N * D * H * W
            img_pc = img_pc.reshape(Nprime, C)

            # Align the geometry to the voxel grid
            geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
            geom_feats = geom_feats.view(Nprime, 3)
            batch_ix = torch.cat(
                [
                    torch.full([Nprime // B, 1], ix, device=img_pc.device, dtype=torch.long)
                    for ix in range(B)
                ]
            )   
            geom_feats = torch.cat([geom_feats, batch_ix], 1)

            # filter out points that are outside box
            kept = (
                (geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2])
            )
            img_pc = img_pc[kept]
            geom_feats = geom_feats[kept]
            try:
                cam_bev = bev_pool(img_pc, geom_feats, B, self.nx[2], self.nx[0], self.nx[1]) # (B, out_channels, nx[2], nx[0], nx[1])
            except:
                cam_bev = torch.zeros(B, C, self.nx[2].item(), self.nx[0].item(), self.nx[1].item(), device = img_pc.device, dtype = img_pc.dtype)

            binary_bev = torch.max(cam_bev, dim=2)[0]

            binary_bev = (binary_bev != 0).float()

        return binary_bev

    def forward(self, 
                cam2ego_T,
                cam_intrins,
                post_cam2ego_T,
                imgs
                ):
        """
        Args:
            cam2ego_T: (B, N, 4, 4), transformation matrix from camera to ego.
            cam_intrins: (B, 3, 3), camera intrinsic matrix.
            post_cam2ego_T: (B, N, 4, 4), transformation matrix from camera to ego after data aug.
            imgs: (B, N, 3, H, W), original image (with data aug).
        Returns:
            bev_feats: (B, C, H, W), bird eye view features.
            cam_bev_mask: (B, H, W), mask for bev_feats. Noting cam_bev_mask[:, i, :, :] is the same for all i.
        """
        cam2ego_rot = cam2ego_T[:, :, :3, :3]
        cam2ego_trans = cam2ego_T[:, :, :3, 3]
        post_cam2ego_rot = post_cam2ego_T[:, :, :3, :3]
        post_cam2ego_trans = post_cam2ego_T[:, :, :3, 3]

        imgs = (imgs - self.mean) / self.std
        img_feats = self.CamEncode(imgs) 
        lss_output = self.lss(
            cam2ego_rot=cam2ego_rot,
            cam2ego_trans=cam2ego_trans,
            cam_intrins=cam_intrins,
            post_cam2ego_rot=post_cam2ego_rot,
            post_cam2ego_trans=post_cam2ego_trans,
            img_feats=img_feats,
        )

        geometry = lss_output["geometry"]
        img_depth_feature = lss_output["img_depth_feature"]
        bev_feats = lss_output["bev_features"]
        with torch.no_grad():
            geom_detach = geometry.detach()
            ones_feat = torch.ones_like(img_depth_feature).to(img_depth_feature.device).detach()
            ref_feats = self.ref_bev_pool(geometry=geom_detach, img_depth_feature=ones_feat)
        B, C, H, W = bev_feats.shape
        cam_bev_mask = ref_feats != 0
        ref_mask = cam_bev_mask[:, 0:1, :, :]
        try:
            assert torch.all(cam_bev_mask == ref_mask) # The masks should be same for different channels.
        except:
            print(f"ERROR: cam_bev_mask != ref_mask")
            torch.save(cam2ego_T, "cam2ego_T.pt")
            torch.save(cam_intrins, "cam_intrins.pt")
            torch.save(post_cam2ego_T, "post_cam2ego_T.pt")
            torch.save(imgs, "imgs.pt")

        bev_feats = bev_feats.permute(0, 2, 3, 1).reshape(B*H*W, C)
        bev_feats = self.proj_head(bev_feats)
        bev_feats = bev_feats.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return bev_feats, cam_bev_mask[:, 0, :, :]
    
def generate_random_rt_matrix(batch_size=1, r_range=(-torch.pi, torch.pi), t_range=(-1, 1)):
    rx = torch.rand(batch_size, 1) * (r_range[1] - r_range[0]) + r_range[0]
    ry = torch.rand(batch_size, 1) * (r_range[1] - r_range[0]) + r_range[0]
    rz = torch.rand(batch_size, 1) * (r_range[1] - r_range[0]) + r_range[0]
    
    zeros = torch.zeros(batch_size, 1)
    ones = torch.ones(batch_size, 1)
    
    Rx = torch.stack([
        torch.cat([ones, zeros, zeros], dim=1),
        torch.cat([zeros, torch.cos(rx), -torch.sin(rx)], dim=1),
        torch.cat([zeros, torch.sin(rx), torch.cos(rx)], dim=1)
    ], dim=2)
    
    Ry = torch.stack([
        torch.cat([torch.cos(ry), zeros, torch.sin(ry)], dim=1),
        torch.cat([zeros, ones, zeros], dim=1),
        torch.cat([-torch.sin(ry), zeros, torch.cos(ry)], dim=1)
    ], dim=2)
    
    Rz = torch.stack([
        torch.cat([torch.cos(rz), -torch.sin(rz), zeros], dim=1),
        torch.cat([torch.sin(rz), torch.cos(rz), zeros], dim=1),
        torch.cat([zeros, zeros, ones], dim=1)
    ], dim=2)
    
    R = torch.bmm(torch.bmm(Rz, Ry), Rx)
    t = torch.rand(batch_size, 3, 1) * (t_range[1] - t_range[0]) + t_range[0]
    
    RT = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, 3:4] = t
    
    return RT.unsqueeze(1)

def test():
    B = 1
    N = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam2ego_T = generate_random_rt_matrix(r_range=(-torch.pi/4, torch.pi/4), t_range=(-1, 1)).to(device)
    cam_intrinsic = torch.tensor([[1.7489e+03, 0.0000e+00, 9.8110e+02],
        [0.0000e+00, 1.7556e+03, 5.4997e+02],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]]).to(device)
    post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(B, N, 1, 1).float().to(device)
    imgs = torch.randn(B, N, 3, 400, 640).to(device)
    imgs = imgs / imgs.max() * 255
    model = Cam2BEV().to(device)
    bev_feats, cam_bev_mask = model(cam2ego_T=cam2ego_T, cam_intrins=cam_intrinsic, post_cam2ego_T=post_cam2ego_T, imgs=imgs)

if __name__ == "__main__":
    for i in range(1000):
        try:
            if i % 10 == 0:
                print(f"iter {i}")
            test()
        except:
            break