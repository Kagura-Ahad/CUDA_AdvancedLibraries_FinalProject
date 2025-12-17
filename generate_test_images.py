#This file generates a sample test image for the GPU Image Processing project
from PIL import Image, ImageDraw
import numpy as np
from typing import cast

#This function creates a colorful gradient image with various shapes
def create_gradient_image(width: int, height: int) -> Image.Image:
    
    #Create an image with RGB mode
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    #Create a gradient background
    for y in range(height):
        #Calculate color based on position
        r = int(255 * (y / height))
        g = int(255 * (1 - y / height))
        b = int(128 + 127 * np.sin(2 * np.pi * y / height))
        
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    #Add some geometric shapes for edge detection testing
    #Circle
    draw.ellipse([300, 200, 600, 500], outline='white', width=5, fill='yellow')
    
    #Rectangle
    draw.rectangle([800, 300, 1200, 700], outline='white', width=5, fill='cyan')
    
    #Triangle
    draw.polygon([1400, 800, 1600, 400, 1800, 800], outline='white', fill='magenta')
    
    #Add text
    try:
        #Try to add text (may fail if font not available)
        draw.text((width//2 - 200, 50), "GPU Image Processing Test", 
                 fill='white')
        draw.text((width//2 - 250, 950), "CUDA at Scale - Capstone Project", 
                 fill='white')
    except:
        pass
    
    return img

#This function creates a more detailed test image with various features
def create_detailed_test_image(width: int = 1920, height: int = 1080) -> Image.Image:
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)
    
    #Background with radial gradient
    center_x, center_y = width // 2, height // 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    
    for x in range(width):
        for y in range(height):
            #Calculate distance from center
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            #Create radial gradient
            intensity = int(255 * (1 - dist / max_radius))
            img.putpixel((x, y), (intensity // 2, intensity // 3, intensity))
    
    draw = ImageDraw.Draw(img)
    
    #Add various shapes for testing different operations
    #Circles
    for i in range(5):
        x = 200 + i * 300
        y = 200 + i * 100
        radius = 80 + i * 20
        color = ((i * 50) % 255, (255 - i * 50) % 255, (i * 80) % 255)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                    fill=color, outline='white', width=3)
    
    #Rectangles
    for i in range(4):
        x1 = 100 + i * 400
        y1 = 600
        x2 = x1 + 250
        y2 = y1 + 200
        color = ((i * 60) % 255, (i * 90) % 255, (255 - i * 60) % 255)
        draw.rectangle([x1, y1, x2, y2], fill=color, outline='yellow', width=3)
    
    #Add some noise for texture
    pixels = img.load()
    for _ in range(5000):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        noise = np.random.randint(-30, 30)
        current_color: tuple[int, int, int] = cast( tuple[int, int, int], pixels[x, y])
        r, g, b = current_color
        pixels[x, y] = (
            max(0, min(255, r + noise)),
            max(0, min(255, g + noise)),
            max(0, min(255, b + noise))
        )
    
    return img

def main():
    print("Generating sample test images...")
    
    #Create and save the gradient image
    img1 = create_gradient_image(1920, 1080)
    img1.save("data/sample_image_1.jpg", quality=95)
    print("Created: data/sample_image_1.jpg")
    
    #Create and save a smaller version for faster testing
    img2 = create_gradient_image(800, 600)
    img2.save("data/sample_image_small.jpg", quality=95)
    print("Created: data/sample_image_small.jpg")
    
    #Create detailed test image
    img3 = create_detailed_test_image(1920, 1080)
    img3.save("data/sample_image_detailed.jpg", quality=95)
    print("Created: data/sample_image_detailed.jpg")
    
if __name__ == "__main__":
    main()