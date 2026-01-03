"""
Compatibility layer for pytorch3d functions.

Provides fallback implementations when pytorch3d is not available.
This allows running on Mac M1/M2 where pytorch3d is hard to install.
"""

import torch
import numpy as np

try:
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.transforms import euler_angles_to_matrix
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False

    def euler_angles_to_matrix(euler_angles, convention="XYZ"):
        """
        Convert rotations given as Euler angles in radians to rotation matrices.

        Args:
            euler_angles: Euler angles in radians, shape (..., 3)
            convention: Euler angle convention (default "XYZ")

        Returns:
            Rotation matrices, shape (..., 3, 3)
        """
        # Implementation based on scipy's rotation matrices
        # For XYZ convention: R = Rz @ Ry @ Rx

        if isinstance(euler_angles, torch.Tensor):
            device = euler_angles.device
            dtype = euler_angles.dtype

            # Get individual angles
            if euler_angles.dim() == 1:
                euler_angles = euler_angles.unsqueeze(0)

            original_shape = euler_angles.shape[:-1]
            euler_angles = euler_angles.reshape(-1, 3)

            if convention == "XYZ":
                x, y, z = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
            else:
                raise ValueError(f"Convention {convention} not implemented")

            cos_x, sin_x = torch.cos(x), torch.sin(x)
            cos_y, sin_y = torch.cos(y), torch.sin(y)
            cos_z, sin_z = torch.cos(z), torch.sin(z)

            # Build rotation matrix R = Rz @ Ry @ Rx
            # Row 0
            r00 = cos_y * cos_z
            r01 = sin_x * sin_y * cos_z - cos_x * sin_z
            r02 = cos_x * sin_y * cos_z + sin_x * sin_z
            # Row 1
            r10 = cos_y * sin_z
            r11 = sin_x * sin_y * sin_z + cos_x * cos_z
            r12 = cos_x * sin_y * sin_z - sin_x * cos_z
            # Row 2
            r20 = -sin_y
            r21 = sin_x * cos_y
            r22 = cos_x * cos_y

            matrix = torch.stack([
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ], dim=-2)

            # Reshape back to original batch shape
            return matrix.reshape(*original_shape, 3, 3)
        else:
            raise TypeError("Expected torch.Tensor")

    def chamfer_distance(x, y, single_directional=False):
        """
        Compute chamfer distance between two point clouds.

        Args:
            x: Point cloud 1, shape (B, N, 3)
            y: Point cloud 2, shape (B, M, 3)
            single_directional: If True, only compute x->y distance

        Returns:
            Tuple of (loss, None)
        """
        # x: (B, N, 3), y: (B, M, 3)
        B, N, _ = x.shape
        _, M, _ = y.shape

        # Compute pairwise distances
        # (B, N, 1, 3) - (B, 1, M, 3) = (B, N, M, 3)
        diff = x.unsqueeze(2) - y.unsqueeze(1)
        dist = (diff ** 2).sum(dim=-1)  # (B, N, M)

        # For each point in x, find nearest in y
        min_dist_x_to_y = dist.min(dim=2)[0]  # (B, N)
        loss_x_to_y = min_dist_x_to_y.mean()

        if single_directional:
            return loss_x_to_y, None

        # For each point in y, find nearest in x
        min_dist_y_to_x = dist.min(dim=1)[0]  # (B, M)
        loss_y_to_x = min_dist_y_to_x.mean()

        return (loss_x_to_y + loss_y_to_x) / 2, None
