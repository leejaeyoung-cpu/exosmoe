import numpy as np
import cv2
from cellpose import models
import torch
import os

class FusionPreprocessor:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        # Initialize Cellpose model (cyto2 is good for general cells)
        # Use CellposeModel directly for newer versions (returns 3 values)
        self.model = models.CellposeModel(gpu=self.use_gpu, model_type='cyto2')
        
    def process_image(self, image_path, save_dir="data/fusion_processed"):
        """
        Reads an image, runs Cellpose, and creates a 3-channel fused tensor.
        Channel 0: Normalized Raw Image
        Channel 1: Cell Masks
        Channel 2: Cell Outlines/Edges
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 1. Read Image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Resize if too large (optional, but good for speed)
        # img = cv2.resize(img, (512, 512))
        
        # 2. Run Cellpose
        # channels=[0,0] means grayscale
        results = self.model.eval([img], diameter=None, channels=[0,0])
        
        if len(results) == 4:
            masks, flows, styles, diams = results
        else:
            masks, flows, styles = results
            
        mask = masks[0]
        
        # 3. Create Channels
        
        # Ch0: Raw Image (Normalized 0-1)
        ch0 = img.astype(np.float32) / 255.0
        
        # Ch1: Mask (Binary or Label) - Let's use Binary for "Cell vs BG" attention
        ch1 = (mask > 0).astype(np.float32)
        
        # Ch2: Edges (Outlines) - Good for shape recognition
        edges = cv2.Canny(mask.astype(np.uint8), 0, 1)
        ch2 = (edges > 0).astype(np.float32)
        
        # Stack: (H, W, 3)
        fused_data = np.dstack((ch0, ch1, ch2))
        
        # 4. Save
        base_name = os.path.basename(image_path)
        name_no_ext = os.path.splitext(base_name)[0]
        save_path = os.path.join(save_dir, f"{name_no_ext}_fused.npy")
        
        np.save(save_path, fused_data)
        
        return save_path, fused_data, mask

if __name__ == "__main__":
    # Test run
    # Assuming the user uploaded image is somewhere, let's pick a dummy path or wait for UI
    pass
