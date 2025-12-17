#This file demonstrates GPU acceleration for image processing tasks including Gaussian blur, edge detection, and image sharpening.
import torch
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import time
import os
import sys
from typing import Any


#This function loads an image and converts it to a numpy array
def load_image(image_path: str) -> NDArray[np.float32]:
    img = Image.open(image_path)
    #Convert to RGB if it is not already in that mode.
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img, dtype=np.float32)


#This function saves a numpy array as an image file
def save_image(image_array: NDArray[Any], output_path: str) -> None:
    #Clip values to valid range and convert to uint8 as required by PIL typehinting
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(image_array)
    img.save(output_path)
    print(f"Saved: {output_path}")


#This function creates a Gaussian kernel for blur operations
def create_gaussian_kernel(size: int, sigma: float = 1.0, device: str | torch.device = 'cuda') -> torch.Tensor:
    ax = torch.arange(-size // 2 + 1., size // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / torch.sum(kernel)

#This function applies a 2D convolution using GPU
def apply_convolution_gpu(image_gpu: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    
    #Reshape image from (H, W, C) to (1, C, H, W) for conv2d
    image_transposed = image_gpu.permute(2, 0, 1).unsqueeze(0)
    
    #Reshape kernel to (C, 1, K, K) for depthwise convolution
    channels = image_transposed.shape[1]
    kernel_expanded = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    
    #Apply convolution with padding
    padding = kernel.shape[0] // 2
    output = torch.nn.functional.conv2d(
        image_transposed, 
        kernel_expanded, 
        padding=padding,
        groups=channels  #Depthwise convolution (separate kernel per channel)
    )
    
    #Reshape back to (H, W, C)
    output = output.squeeze(0).permute(1, 2, 0)
    
    return output


#This function applies Gaussian blur using GPU
def gaussian_blur_gpu(image_gpu: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    device: torch.device | str = image_gpu.device
    kernel = create_gaussian_kernel(kernel_size, sigma, device=device)
    return apply_convolution_gpu(image_gpu, kernel)


#This function applies Sobel edge detection using GPU
def edge_detection_gpu(image_gpu: torch.Tensor) -> torch.Tensor:
    
    #Get device from input tensor
    device: torch.device = image_gpu.device
    
    #These are the sobel kernels for x and y directions
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=device)
    
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=device)
    
    #Convert to grayscale
    gray = torch.mean(image_gpu, dim=2, keepdim=True)
    gray = gray.repeat(1, 1, 3)
    
    #Apply Sobel filters
    grad_x = apply_convolution_gpu(gray, sobel_x)
    grad_y = apply_convolution_gpu(gray, sobel_y)
    
    #Compute magnitude
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    return magnitude


#This function applies sharpening filter using GPU
def sharpen_gpu(image_gpu: torch.Tensor) -> torch.Tensor:

    #Get device from input tensor
    device: torch.device = image_gpu.device

    #Sharpening kernel
    sharpen_kernel = torch.tensor([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]], dtype=torch.float32, device=device)
    
    return apply_convolution_gpu(image_gpu, sharpen_kernel)


#This function adjusts brightness and contrast using GPU
def brightness_contrast_gpu(image_gpu: torch.Tensor, brightness: float = 0, contrast: float = 1.0) -> torch.Tensor:
    adjusted = image_gpu * contrast + brightness
    return torch.clamp(adjusted, 0, 255)

#This function processes an image with specified operations using GPU acceleration
def process_image(input_path: str, output_dir: str, operations: list[str]) -> dict[str, str]:
    #Load image
    print("Loading image...")
    image_np: NDArray[np.float32] = load_image(input_path)
    print(f"Image shape: {image_np.shape}")
    print(f"Image size: {image_np.shape[0] * image_np.shape[1] * image_np.shape[2] * 4 / (1024**2):.2f} MB")
    
    #Transfer to GPU
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_gpu: torch.Tensor = torch.from_numpy(image_np).to(device) #type: ignore
    
    #Get base filename
    base_name: str = os.path.splitext(os.path.basename(input_path))[0]
    
    results: dict[str, str] = {}
    
    #Apply operations
    for op in operations:
        print(f"\nApplying {op}...")
        start_time: float = time.time()
        
        result_gpu: torch.Tensor
        output_name: str
        if op == 'blur':
            result_gpu = gaussian_blur_gpu(image_gpu, kernel_size=9, sigma=2.0)
            output_name = f"{base_name}_blurred.png"
        elif op == 'edge':
            result_gpu = edge_detection_gpu(image_gpu)
            output_name = f"{base_name}_edges.png"
        elif op == 'sharpen':
            result_gpu = sharpen_gpu(image_gpu)
            output_name = f"{base_name}_sharpened.png"
        elif op == 'brighten':
            result_gpu = brightness_contrast_gpu(image_gpu, brightness=30, contrast=1.2)
            output_name = f"{base_name}_brightened.png"
        else:
            print(f"Unknown operation: {op}")
            continue
        
        #Synchronize GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed: float = time.time() - start_time
        
        #Transfer back to CPU and save
        result_np: NDArray[Any] = result_gpu.cpu().numpy() #type: ignore
        output_path: str = os.path.join(output_dir, output_name)
        save_image(result_np, output_path)
        
        print(f"GPU processing time: {elapsed:.4f} seconds")
        results[op] = output_path
    
    return results


def main() -> None:
    input: str = "data/sample_image_small.jpg"
    output: str = "output"
    operations: list[str] = ['blur', 'edge', 'sharpen', 'brighten']

    #Check if input file exists
    if not os.path.exists(input):
        print(f"Error: Input file '{input}' not found!")
        sys.exit(1)
    
    #Create output directory
    os.makedirs(output, exist_ok=True)
    
    #Process image
    process_image(input, output, operations)
    
    print(f"Task completed and output files saved to: {output}")


if __name__ == "__main__":
    main()
