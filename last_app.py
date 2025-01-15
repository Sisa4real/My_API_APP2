import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Path to the model
MODEL_PATH = "CropDisease_model3.tflite"

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path):
    """
    Preprocesses an image to match the model's input format.
    - Resizes to (128, 128).
    - Normalizes pixel values to [0, 1].
    - Converts to float32 and adds batch dimension.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    input_data = np.expand_dims(image_array, axis=0).astype(np.float32)
    return input_data

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to get predictions for a single image.
    Expects an uploaded image file.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    # Save the uploaded image to a temporary file
    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)

    # Preprocess the image
    input_data = preprocess_image(temp_image_path)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index']).flatten()
    predicted_class_index = int(np.argmax(output_data))
    confidence = float(np.max(output_data))

    # Remove the temporary image
    os.remove(temp_image_path)

    return jsonify({
        "predicted_class_index": predicted_class_index,
        "confidence": f"{confidence:.2%}",
        "probabilities": [float(prob) for prob in output_data]
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
