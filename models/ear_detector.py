#file models/ear_detector.py
import cv2
import numpy as np
import os
from PIL import Image

class SimpleEarDetector:
    """Simple ear region extractor focusing on upper portion of elephant images"""

    def __init__(self, ear_region_ratio=0.4):
        self.ear_region_ratio = ear_region_ratio

    def extract_ear_region(self, image_path):
        """Extract ear region from elephant image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Extract upper portion where ears are located
        ear_height = int(h * self.ear_region_ratio)
        ear_region = image[:ear_height, :]

        # Enhance contrast for better ear visibility
        ear_region = self._enhance_image(ear_region)

        return ear_region

    def _enhance_image(self, image):
        """Enhance image contrast and sharpness"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)

        # Merge channels and convert back to RGB
        lab = cv2.merge((l_channel, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return enhanced

    def process_dataset(self, data_dir, output_dir):
        """Process entire dataset and save cropped ear regions"""
        os.makedirs(output_dir, exist_ok=True)

        for elephant_folder in os.listdir(data_dir):
            elephant_path = os.path.join(data_dir, elephant_folder)
            if not os.path.isdir(elephant_path):
                continue

            output_elephant_dir = os.path.join(output_dir, elephant_folder)
            os.makedirs(output_elephant_dir, exist_ok=True)

            for img_file in os.listdir(elephant_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(elephant_path, img_file)
                    ear_region = self.extract_ear_region(img_path)

                    if ear_region is not None:
                        output_path = os.path.join(output_elephant_dir, img_file)
                        Image.fromarray(ear_region).save(output_path)

            print(f"Processed {elephant_folder}")

        print(f"All ear regions saved to {output_dir}")
