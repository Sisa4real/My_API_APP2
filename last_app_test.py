import os
import requests

# API URL
API_URL = "http://127.0.0.1:5000/predict"

# Path to the folder containing test images
TEST_IMAGES_DIR = "Test"

# Get all test image file paths
image_files = [
    os.path.join(TEST_IMAGES_DIR, f)
    for f in os.listdir(TEST_IMAGES_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

if not image_files:
    print("No images found in the test directory!")
else:
    for idx, image_file in enumerate(sorted(image_files)):
        print(f"Testing image {idx + 1}/{len(image_files)}: {image_file}")

        # Open the image file in binary mode
        with open(image_file, 'rb') as img:
            # Send the image to the API
            response = requests.post(API_URL, files={"image": img})

        if response.status_code == 200:
            result = response.json()
            print("Class Probabilities:")
            for i, prob in enumerate(result['probabilities']):
                print(f"  Class {i}: {prob:.4f}")

            print(f"Predicted Class Index: {result['predicted_class_index']}")
            print(f"Confidence: {result['confidence']}")
        else:
            print(f"Failed to get prediction for {image_file}")
            print(f"Error: {response.text}")

        print("-" * 40)
