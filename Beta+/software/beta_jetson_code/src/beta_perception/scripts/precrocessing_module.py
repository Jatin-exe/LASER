import torch
import torch.nn.functional as F

class PreprocessingModule(torch.nn.Module):
    """
    A PyTorch module for preprocessing input images. Includes:
    - Conversion from BGR to RGB
    - Normalization to [0, 1]
    - Resizing to fixed dimensions (384x640)
    - Conversion to FP16 precision
    """

    def __init__(self):
        super(PreprocessingModule, self).__init__()

    def forward(self, image: torch.Tensor):
        """
        Preprocesses the input image tensor.

        Args:
            image (torch.Tensor): Input image tensor in BGR format with shape (H, W, 3).
                                  Expected data type: torch.uint8.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Preprocessed image tensor in RGB format 
            with shape (3, 384, 640) and data type torch.float16, original height, and original width.
        """
        # Step 1: Get original dimensions
        original_height, original_width = image.shape[0], image.shape[1]

        # Convert original dimensions to GPU tensors
        original_height_tensor = torch.tensor([original_height], dtype=torch.int32, device=image.device)
        original_width_tensor = torch.tensor([original_width], dtype=torch.int32, device=image.device)

        # Step 2: Convert from BGR to RGB explicitly
        image = image[:, :, [2, 1, 0]]  # Specify channel reordering explicitly

        # Step 3: Normalize to [0, 1] and convert to FP16
        image = image.permute(2, 0, 1).to(dtype=torch.float16) / 255.0  # Convert HWC to CHW

        # Step 4: Resize to fixed dimensions (384x640)
        image = F.interpolate(
            image.unsqueeze(0), size=(384, 640), mode="bilinear", align_corners=False
        ) 

        return image.cpu(), original_height_tensor.cpu(), original_width_tensor.cpu()

# Example usage
if __name__ == "__main__":
    # Create the module instance
    module = PreprocessingModule().to('cuda:0')

    # Example input tensor (HWC format, BGR, uint8)
    example_image = torch.randint(0, 256, (1216, 1936, 3), dtype=torch.uint8).to('cuda:0')

    # Preprocess the image
    preprocessed_image, height_tensor, width_tensor = module(example_image)
    print("Preprocessed Image Shape:", preprocessed_image.shape)
    print("Preprocessed Image Data Type:", preprocessed_image.dtype)
    print("Original Height:", height_tensor.item())
    print("Original Width:", width_tensor.item())

    # Trace the module
    print("Tracing the module...")
    traced_module = torch.jit.script(module)

    # Save the traced module
    traced_module.save("model.pt")
    print("Traced module saved as 'model.pt'")
