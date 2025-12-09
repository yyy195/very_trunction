from typing import List
import numpy as np
from PIL import Image
import random  

def image_augment_from_PIL(image):

        if isinstance(image,List):
            for img in image:
                noise_level = random.randint(1,40)
                img = add_image_noise(img,noise_level=noise_level)
            return image
        
        elif isinstance(image,Image.Image):
            noise_level = random.randint(1,40)
            image = add_image_noise(image,noise_level=noise_level)

            return image
        
        else:
            raise ValueError("Invalid image format, must be a list of PIL or a PIL")

def add_image_noise(image, noise_level=10, noise_type='gaussian'):
    
    # 1. read image and convert to RGB mode
    if not isinstance(image,Image.Image):
        img = Image.open(image).convert('RGB')
    else:
        img = image
    img_array = np.array(img).astype(np.float32)
    
    # 2. generate noise
    noise = None
    max_pixel = 255.0
    scaled_noise_level = noise_level * max_pixel / 100
    
    if noise_type == 'gaussian':
        noise = np.random.normal(
            loc=0, 
            scale=scaled_noise_level, 
            size=img_array.shape
        )
    elif noise_type == 'salt_pepper':
        salt_pepper = np.random.choice(
            [0, 1, 2],  
            size=img_array.shape[:2],
            p=[
                1 - (noise_level/100),  
                (noise_level/100)/2,     
                (noise_level/100)/2      
            ]
        )
        
        noise = np.zeros_like(img_array)
        noise[salt_pepper == 1] = max_pixel  
        noise[salt_pepper == 2] = -max_pixel  
    else:
        raise ValueError(f"unsupported noise type: {noise_type}")
    
    # 3. add noise and limit pixel range
    noisy_array = img_array + noise
    noisy_array = np.clip(noisy_array, 0, max_pixel).astype(np.uint8)
    
    # 4. convert back to PIL image
    noisy_img = Image.fromarray(noisy_array)
    
        
    return noisy_img