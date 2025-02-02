import cv2
import numpy as np
import tensorflow as tf
import argparse
import sys

# Command-line argument parsing:
parser = argparse.ArgumentParser(description="TFLite Object Detection on an Image")
parser.add_argument('--image', type=str, required=True, help='Path to the input image')
args = parser.parse_args()

# Paths to your model and label files.
MODEL_PATH = "/root/MultiAgent/FTCTraining/final_output/limelight_neural_detector_8bit.tflite"
LABELS_PATH = "/root/MultiAgent/FTCTraining/final_output/limelight_neural_detector_labels.txt"

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(LABELS_PATH)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Determine the input size (assuming shape [1, height, width, channels]).
input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]

# Load the image.
image = cv2.imread(args.image)
if image is None:
    print(f"Error: Could not load image from {args.image}")
    sys.exit(1)
original_image = image.copy()
img_height, img_width, _ = image.shape

# Preprocess the image: resize and add batch dimension.
resized = cv2.resize(image, (input_width, input_height))
input_data = np.expand_dims(resized, axis=0)
if input_details[0]['dtype'] == np.float32:
    input_data = np.float32(input_data) / 255.0

# Run inference.
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Extract outputs.
# Typically for SSD MobileNet V2 the outputs are:
#   detection_boxes: [1, num_detections, 4]
#   detection_classes: [1, num_detections]
#   detection_scores: [1, num_detections]
#   num_detections: [1]
boxes = interpreter.get_tensor(output_details[0]['index'])
classes = interpreter.get_tensor(output_details[1]['index'])
scores = interpreter.get_tensor(output_details[2]['index'])
num_detections_tensor = interpreter.get_tensor(output_details[3]['index'])

# Remove batch dimension and force outputs into proper array shapes.
boxes = np.squeeze(boxes)
if boxes.ndim == 1:
    boxes = np.expand_dims(boxes, axis=0)
classes = np.squeeze(classes)
scores = np.squeeze(scores)

# Ensure scores and classes are at least 1D arrays.
scores = np.atleast_1d(scores)
classes = np.atleast_1d(classes)

# Determine number of detections.
try:
    num_detections = int(num_detections_tensor[0])
except Exception:
    num_detections = boxes.shape[0]

# Set a confidence threshold.
confidence_threshold = 0.5

for i in range(num_detections):
    if i >= len(scores):
        break

    score = scores[i]
    if score < confidence_threshold:
        continue

    # Get the detection box.
    box = boxes[i]
    if len(box) > 4:
        box = box[:4]
    # Unpack the normalized coordinates.
    ymin, xmin, ymax, xmax = box

    # Convert normalized coordinates to pixel values.
    x1 = int(xmin * img_width)
    y1 = int(ymin * img_height)
    x2 = int(xmax * img_width)
    y2 = int(ymax * img_height)

    # Process the class value:
    # Convert classes[i] to a NumPy array, flatten it, and use the first element.
    class_val = np.array(classes[i]).flatten()
    class_index = int(class_val[0])
    
    label = labels[class_index] if class_index < len(labels) else f"Class {class_index}"
    label_text = f"{label}: {score:.2f}"

    # Draw the bounding box.
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Compute text size and draw a filled rectangle for readability.
    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(original_image, (x1, y1 - text_height - baseline),
                  (x1 + text_width, y1), (0, 255, 0), cv2.FILLED)
    cv2.putText(original_image, label_text, (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Display the resulting image with detections.
cv2.imshow("Object Detection", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
