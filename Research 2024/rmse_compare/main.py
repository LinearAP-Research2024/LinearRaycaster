import cv2
import numpy as np

def rmse(image1, image2):
    """Compute Root Mean Squared Error (RMSE) between two images."""
    return np.sqrt(np.mean((image1 - image2) ** 2))

def load_image(file_path):
    """Load an image from file."""
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def main(image_path1, image_path2):
    # Load images
    img1 = load_image(image_path1)
    img2 = load_image(image_path2)

    if img1 is None or img2 is None:
        print("Failed to load one or both images.")
        return

    # Resize images to the same dimensions (for images of different sizes)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Compute RMSE
    error = rmse(img1, img2)

    print(f"RMSE between {image_path1} and {image_path2}: {error}")

if __name__ == "__main__":
    # Example usage:
    image_path1 = "recolor128.png"  # File path of first image
    image_path2 = "SheppLogan_Phantom.png"  # File path of second image
    main(image_path1, image_path2)
    
    image_path1 = "recolor256.png"  # File path of first image
    image_path2 = "SheppLogan_Phantom.png"  # File path of second image
    main(image_path1, image_path2)
    
    image_path1 = "recolor512.png"  # File path of first image
    image_path2 = "SheppLogan_Phantom.png"  # File path of second image
    main(image_path1, image_path2)
    
    image_path1 = "recolor1024.png"  # File path of first image
    image_path2 = "SheppLogan_Phantom.png"  # File path of second image
    main(image_path1, image_path2)