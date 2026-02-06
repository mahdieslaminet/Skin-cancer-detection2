import os
import shutil
import random
from pathlib import Path

def get_sample_image():
    """Extract a random sample image from the downloaded dataset"""
    
    # Path to the dataset
    dataset_path = os.path.expanduser("~/.cache/kagglehub/datasets/nodoubttome/skin-cancer9-classesisic/versions/1")
    dataset_dir = os.path.join(dataset_path, 'Skin cancer ISIC The International Skin Imaging Collaboration')
    train_dir = os.path.join(dataset_dir, 'Train')
    
    if not os.path.exists(train_dir):
        print(f"Dataset not found at {train_dir}")
        print("Please make sure you've downloaded the dataset using the notebook.")
        return None
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    if not class_dirs:
        print("No class directories found in the dataset.")
        return None
    
    # Pick a random class
    random_class = random.choice(class_dirs)
    class_path = os.path.join(train_dir, random_class)
    
    # Get all images in that class
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = []
    for file in os.listdir(class_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            images.append(os.path.join(class_path, file))
    
    if not images:
        print(f"No images found in {random_class}")
        return None
    
    # Pick a random image
    random_image = random.choice(images)
    
    # Create test_images directory
    test_dir = 'test_images'
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy the image to test_images directory
    dest_path = os.path.join(test_dir, f'sample_{random_class}_{os.path.basename(random_image)}')
    shutil.copy2(random_image, dest_path)
    
    print(f"Sample image extracted!")
    print(f"Class: {random_class}")
    print(f"Source: {random_image}")
    print(f"Copied to: {dest_path}")
    print(f"\nYou can now use this image to test the API:")
    print(f"  - Open http://localhost:5001 in your browser")
    print(f"  - Upload the image from: {dest_path}")
    
    return dest_path

if __name__ == '__main__':
    get_sample_image()

