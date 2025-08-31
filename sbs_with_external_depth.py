
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import tqdm
from comfy.utils import ProgressBar

class SBS_External_Depthmap_by_SamSeen:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "depth_scale": ("INT", {"default": 30}),
                "mode": (["Parallel", "Cross-eyed"], {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stereoscopic_image",)
    FUNCTION = "process"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "Legacy version: Create side-by-side (SBS) stereoscopic images and videos using your own custom depth maps. For advanced users who want complete control over the 3D effect."

    def process(self, base_image, depth_map, depth_scale, mode="Cross-eyed"):
        """
        Create a side-by-side (SBS) stereoscopic image from a standard image or batch.

        Parameters:
        - base_image: tensor representing the base image(s) [B, H, W, C].
        - depth_map: tensor representing the depth map(s) [B, H', W', C], [B, H', W', 1], or [B, H', W'].
        - depth_scale: integer representing the scaling factor for depth.
        - mode: "Parallel" or "Cross-eyed" viewing mode.

        Returns:
        - sbs_image: the stereoscopic image(s) [B, H, W*2, C].
        """
        # Log initial shapes
        print(f"Initial shapes: base_image={getattr(base_image, 'shape', None)}, depth_map={getattr(depth_map, 'shape', None)}")

        # Validate input dimensions
        if len(base_image.shape) not in [3, 4] or len(depth_map.shape) not in [2, 3, 4]:
            raise ValueError(f"Invalid input dimensions. Got base_image: {base_image.shape}, depth_map: {depth_map.shape}")

        # Move tensors to GPU/CPU target
        base_image = base_image.to(self.device).float().contiguous()
        depth_map = depth_map.to(self.device).float().contiguous()

        # Handle batched or non-batched input
        if len(base_image.shape) == 4:  # [B, H, W, C]
            B, H, W, C = base_image.shape
        else:  # [H, W, C]
            B = 1
            H, W, C = base_image.shape
            base_image = base_image.unsqueeze(0)  # Add batch dimension
            # Ensure depth_map also has a batch dim
            if len(depth_map.shape) in [2, 3]:
                depth_map = depth_map.unsqueeze(0)

        # Validate image channels
        if C != 3:
            raise ValueError(f"Base image must have 3 channels (RGB). Got shape: {base_image.shape}")

        # Ensure depth_map has batch dimension
        if len(depth_map.shape) == 3:  # [H', W', C] or [H', W', 1]
            depth_map = depth_map.unsqueeze(0)  # [1, H', W', C]
        elif len(depth_map.shape) == 2:  # [H', W']
            depth_map = depth_map.unsqueeze(0).unsqueeze(-1)  # [1, H', W', 1]

        # Validate batch sizes
        if base_image.shape[0] != depth_map.shape[0]:
            raise ValueError(f"Batch sizes must match. Got base_image: {base_image.shape}, depth_map: {depth_map.shape}")

        # Clamp base image to [0, 1] just in case
        base_image = base_image.clamp(0, 1)

        # Process each image in the batch
        sbs_images = []
        pbar = ProgressBar(B)

        for b in range(B):
            # Get the current image and depth map
            current_image = base_image[b]  # [H, W, C]
            current_depth_map = depth_map[b]  # [H', W', C] or [H', W', 1]

            # Log initial shapes for this batch
            print(f"Batch {b+1}/{B}: initial image shape={current_image.shape}, initial depth shape={current_depth_map.shape}")

            # Handle transposed or unexpected depth map shapes
            if len(current_depth_map.shape) == 3:
                if current_depth_map.shape[2] in [1, 3]:
                    depth_for_sbs = current_depth_map[:, :, 0]  # Take first channel [H', W']
                else:
                    # Check for transposed shape (e.g., [C, H', W'])
                    if current_depth_map.shape[0] in [1, 3]:
                        print(f"Batch {b+1}/{B}: detected possible transposed depth map, permuting...")
                        current_depth_map = current_depth_map.permute(1, 2, 0)  # [H', W', C]
                        depth_for_sbs = current_depth_map[:, :, 0]  # [H', W']
                    else:
                        raise ValueError(f"Unexpected depth map channels: {current_depth_map.shape}")
            elif len(current_depth_map.shape) == 2:
                depth_for_sbs = current_depth_map  # Already [H', W']
            else:
                raise ValueError(f"Unexpected depth map shape: {current_depth_map.shape}")

            # Log shape after channel extraction
            print(f"Batch {b+1}/{B}: depth shape after channel extraction={depth_for_sbs.shape}")

            # Resize depth map to match image dimensions (use NCHW layout for interpolate)
            if depth_for_sbs.shape != (H, W):
                print(f"Batch {b+1}/{B}: resizing depth map from {depth_for_sbs.shape} to {(H, W)}")
                depth_for_sbs = depth_for_sbs.to(torch.float32).contiguous()
                depth_resized = F.interpolate(
                    depth_for_sbs.unsqueeze(0).unsqueeze(0),  # [1, 1, H', W']  (NCHW)
                    size=(H, W),
                    mode='nearest',
                    align_corners=None
                ).squeeze(0).squeeze(0)  # [H, W]
                depth_for_sbs = depth_resized

            # Validate depth map shape
            if depth_for_sbs.shape != (H, W):
                raise ValueError(
                    f"Depth map dimensions must match image dimensions after resizing. "
                    f"Got depth: {depth_for_sbs.shape}, image: {(H, W)}"
                )

            # Log final depth shape
            print(f"Batch {b+1}/{B}: final depth shape={depth_for_sbs.shape}")

            # Normalize depth to [0,1] safely (avoid divide-by-zero)
            dmin = torch.min(depth_for_sbs)
            dmax = torch.max(depth_for_sbs)
            denom = (dmax - dmin).clamp_min(1e-8)
            depth_for_sbs = (depth_for_sbs - dmin) / denom

            # Create SBS image
            sbs_image = torch.zeros((H, W * 2, 3), device=self.device, dtype=torch.float32)
            # Assign original image based on mode
            if mode == "Cross-eyed":
                sbs_image[:, W:, :] = current_image  # Right view
                sbs_image[:, :W, :] = current_image  # Left view
            else:  # Parallel
                sbs_image[:, :W, :] = current_image  # Left view
                sbs_image[:, W:, :] = current_image  # Right view

            # Compute disparity (shift)
            depth_scaling = depth_scale / max(float(W), 1.0)
            disparity = depth_for_sbs * depth_scaling  # [H, W]

            # Set flip for mode
            fliped = W if mode == "Cross-eyed" else 0

            # Log SBS setup
            print(f"Batch {b+1}/{B}: sbs_image shape={sbs_image.shape}, fliped={fliped}")

            # Create grid for sampling
            y_coords = torch.arange(H, device=self.device)
            x_coords = torch.arange(W, device=self.device)
            y_coords, x_coords = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid = torch.stack([x_coords.float() / max((W - 1), 1), y_coords.float() / max((H - 1), 1)], dim=-1) * 2 - 1  # Normalize to [-1,1]

            # Apply disparity to x grid
            shift_grid = grid.clone()
            shift_grid[:, :, 0] = shift_grid[:, :, 0] + (disparity / max((W - 1), 1) * 2)  # Normalize shift

            # Sample the image with the shifted grid (NCHW expected)
            current_image_shift = current_image.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, C, H, W]
            shifted = F.grid_sample(
                current_image_shift, shift_grid.unsqueeze(0),
                mode='nearest', padding_mode='border', align_corners=True
            ).squeeze(0).permute(1, 2, 0)  # [H, W, C]

            # Apply shifted image based on mode
            sbs_image[:, fliped:fliped + W, :] = shifted

            # Gap filling approximation (limited to max_shift=10 for speed)
            max_shift = 10
            for shift in range(1, max_shift + 1):
                shift_grid = grid.clone()
                shift_grid[:, :, 0] = shift_grid[:, :, 0] + ((disparity + shift) / max((W - 1), 1) * 2)
                shifted_shift = F.grid_sample(
                    current_image_shift, shift_grid.unsqueeze(0),
                    mode='nearest', padding_mode='border', align_corners=True
                ).squeeze(0).permute(1, 2, 0)
                sbs_image[:, fliped:fliped + W, :] = torch.where(
                    shifted_shift > 0, shifted_shift, sbs_image[:, fliped:fliped + W, :]
                )

            # Move result to CPU and add batch dimension
            sbs_images.append(sbs_image.unsqueeze(0).cpu())
            pbar.update(1)

        # Stack the results
        sbs_images_batch = torch.cat(sbs_images, dim=0)

        # Print final output stats
        print(f"Final SBS image batch shape: {sbs_images_batch.shape}, min: {sbs_images_batch.min().item()}, max: {sbs_images_batch.max().item()}")

        return (sbs_images_batch,)
