import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import tqdm
import os
import sys
import cv2
from comfy.utils import ProgressBar

# Add the current directory to the path so we can import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our depth estimation implementation
try:
    from depth_estimator import DepthEstimator
    print("Successfully imported DepthEstimator")
except ImportError as e:
    print(f"Error importing DepthEstimator: {e}")

    # Define a placeholder class that will show a clear error
    class DepthEstimator:
        def __init__(self):
            print("ERROR: DepthEstimator could not be imported!")

        def load_model(self):
            print("ERROR: DepthEstimator model could not be loaded!")
            return None

        def predict_depth(self, image):
            print("ERROR: DepthEstimator model could not be used for inference!")
            # Return a blank depth map
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.float32)

class SBS_V2_by_SamSeen:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = None
        self.original_depths = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "depth_scale": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 100.0, "step": 0.5}),
                "blur_radius": ("INT", {"default": 3, "min": 1, "max": 51, "step": 2}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "mode": (["Parallel", "Cross-eyed"], {"default": "Cross-eyed"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("stereoscopic_image", "depth_map")
    FUNCTION = "process"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "Create stunning side-by-side (SBS) stereoscopic images and videos with automatic depth map generation using Depth-Anything-V2. Perfect for VR content, 3D displays, and image sequences!"

    def load_depth_model(self):
        """
        Load the depth model.
        """
        # Create a new instance of our depth model if needed
        if self.depth_model is None:
            print("Creating new DepthEstimator instance")
            self.depth_model = DepthEstimator()

        # Load the model
        try:
            self.depth_model.load_model()
            print("Successfully loaded DepthEstimator model")
        except Exception as e:
            import traceback
            print(f"Error loading DepthEstimator model: {e}")
            print(traceback.format_exc())

        return self.depth_model

    def generate_depth_map(self, image):
        """
        Generate a depth map from an image or batch of images.
        """
        try:
            # Load the model if not already loaded
            depth_model = self.load_depth_model()

            # Process the image
            B, H, W, C = image.shape
            pbar = ProgressBar(B)
            out = []

            # Store original depth maps for each image in the batch
            self.original_depths = []

            # Process each image in the batch
            for b in range(B):
                # Convert tensor to numpy for processing
                img_np = image[b].cpu().numpy() * 255.0  # Scale to 0-255
                img_np = img_np.astype(np.uint8)

                print(f"Processing image {b+1}/{B} with shape: {img_np.shape}")

                # Use our depth model's predict_depth method
                depth = depth_model.predict_depth(img_np)

                print(f"Raw depth output: shape={depth.shape}, min={np.min(depth)}, max={np.max(depth)}, mean={np.mean(depth)}")

                # Make sure depth is normalized to [0,1]
                if np.min(depth) < 0 or np.max(depth) > 1:
                    depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

                # Save the original depth map for the SBS generation
                self.original_depths.append(depth.copy())

                # Convert back to tensor - keep as grayscale
                # First expand to 3 channels (all with same values) for ComfyUI compatibility
                depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)  # Add channel dimension

                out.append(depth_tensor)
                pbar.update(1)

            # Stack the depth maps
            depth_out = torch.stack(out)

            print(f"Stacked depth maps: shape={depth_out.shape}, min={depth_out.min().item()}, max={depth_out.max().item()}, mean={depth_out.mean().item()}")

            # Make sure it's in the right format for ComfyUI (B,H,W,C)
            # For grayscale, we need to expand to 3 channels for ComfyUI compatibility
            if len(depth_out.shape) == 3:  # [B,1,H,W]
                depth_out = depth_out.permute(0, 2, 3, 1).cpu().float()  # [B,H,W,1]
                depth_out = depth_out.repeat(1, 1, 1, 3)  # [B,H,W,3]
            elif len(depth_out.shape) == 4:  # [B,C,H,W]
                depth_out = depth_out.permute(0, 2, 3, 1).cpu().float()  # [B,H,W,C]

            print(f"Final depth map shape: {depth_out.shape}, min: {depth_out.min().item()}, max: {depth_out.max().item()}, mean: {depth_out.mean().item()}")

            return depth_out
        except Exception as e:
            import traceback
            print(f"Error generating depth map: {e}")
            print(traceback.format_exc())
            # Return a blank depth map in case of error
            B, H, W, C = image.shape
            print(f"Creating blank depth map with shape: {(B, H, W, C)}")
            return torch.zeros((B, H, W, C), dtype=torch.float32)

    def process(self, base_image, depth_scale, blur_radius, invert_depth=False, mode="Cross-eyed"):
        """
        Create a side-by-side (SBS) stereoscopic image from a standard image or image sequence.
        The depth map is automatically generated using our custom depth estimation approach.

        Parameters:
        - base_image: tensor representing the base image(s) with shape [B,H,W,C].
        - depth_scale: float representing the scaling factor for depth.
        - blur_radius: integer controlling the smoothness of the depth map.
        - invert_depth: boolean to invert the depth map (swap foreground/background).
        - mode: "Parallel" or "Cross-eyed" viewing mode.

        Returns:
        - sbs_image: the stereoscopic image(s).
        - depth_map: the generated depth map(s).
        """
        # Update the depth model parameters
        if self.depth_model is not None:
            # Set default edge_weight for compatibility
            self.depth_model.edge_weight = 0.5
            # Keep gradient_weight for compatibility but set to 0
            self.depth_model.gradient_weight = 0.0
            self.depth_model.blur_radius = blur_radius

        # Generate depth map
        print(f"Generating depth map with blur_radius={blur_radius}, invert_depth={invert_depth}...")
        depth_map = self.generate_depth_map(base_image)

        # Get batch size
        B = base_image.shape[0]

        # Move input to GPU
        base_image = base_image.to(self.device)

        # Process each image in the batch
        sbs_images = []
        enhanced_depth_maps = []

        for b in range(B):
            # Get the current image
            current_image = base_image[b]

            # Get the current depth map
            if len(self.original_depths) > b:
                depth_for_sbs = torch.from_numpy(self.original_depths[b]).to(self.device).float()
            else:
                current_depth_map = depth_map[b].to(self.device)
                depth_for_sbs = current_depth_map[:, :, 0] if current_depth_map.shape[2] == 3 else current_depth_map

            # Invert depth if requested
            if invert_depth:
                print("Inverting depth map (swapping foreground/background)")
                depth_for_sbs = 1.0 - depth_for_sbs

            # Get dimensions
            H, W = depth_for_sbs.shape

            # Create SBS image
            sbs_image = torch.zeros((H, W * 2, 3), device=self.device, dtype=torch.float32)
            # Assign original image based on mode
            if mode == "Cross-eyed":
                sbs_image[:, W:, :] = current_image  # Right view
                sbs_image[:, :W, :] = current_image  # Left view
            else:  # Parallel
                sbs_image[:, :W, :] = current_image  # Left view
                sbs_image[:, W:, :] = current_image  # Right view

            # Normalize depth for disparity
            depth_for_sbs = (depth_for_sbs - depth_for_sbs.min()) / (depth_for_sbs.max() - depth_for_sbs.min() + 1e-8)

            # Compute disparity (shift)
            disparity = depth_for_sbs * (depth_scale / W)  # [H, W]

            # Set flip for mode
            fliped = W if mode == "Cross-eyed" else 0

            # Create grid for sampling
            y_coords, x_coords = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
            grid = torch.stack([x_coords.float() / (W - 1), y_coords.float() / (H - 1)], dim=-1) * 2 - 1  # Normalize to [-1,1]

            # Apply disparity to x grid
            shift_grid = grid.clone()
            shift_grid[:, :, 0] = shift_grid[:, :, 0] + (disparity / (W - 1) * 2)  # Normalize shift

            # Sample the image with the shifted grid
            current_image_shift = current_image.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            shifted = F.grid_sample(current_image_shift, shift_grid.unsqueeze(0), mode='nearest', padding_mode='border', align_corners=True).squeeze(0).permute(1, 2, 0)  # [H, W, C]

            # Apply shifted image based on mode
            sbs_image[:, fliped:fliped + W, :] = shifted

            # Gap filling approximation (limited to max_shift=10 for speed)
            max_shift = 10
            for shift in range(1, max_shift + 1):
                shift_grid = grid.clone()
                shift_grid[:, :, 0] = shift_grid[:, :, 0] + ((disparity + shift) / (W - 1) * 2)
                shifted_shift = F.grid_sample(current_image_shift, shift_grid.unsqueeze(0), mode='nearest', padding_mode='border', align_corners=True).squeeze(0).permute(1, 2, 0)
                sbs_image[:, fliped:fliped + W, :] = torch.where(shifted_shift > 0, shifted_shift, sbs_image[:, fliped:fliped + W, :])

            # Move result to CPU for output
            sbs_images.append(sbs_image.cpu())

            # Create 3-channel depth map for output
            depth_gray = depth_for_sbs.cpu().numpy()
            depth_3ch = np.stack([depth_gray, depth_gray, depth_gray], axis=-1)
            enhanced_depth_map = torch.tensor(depth_3ch)

            enhanced_depth_maps.append(enhanced_depth_map)

        # Stack the results
        sbs_images_batch = torch.stack(sbs_images)
        enhanced_depth_maps_batch = torch.stack(enhanced_depth_maps)

        # Print final output stats
        print(f"Final SBS image batch shape: {sbs_images_batch.shape}, min: {sbs_images_batch.min().item()}, max: {sbs_images_batch.max().item()}")
        print(f"Final depth map batch shape: {enhanced_depth_maps_batch.shape}, min: {enhanced_depth_maps_batch.min().item()}, max: {enhanced_depth_maps_batch.max().item()}")

        return (sbs_images_batch, enhanced_depth_maps_batch)