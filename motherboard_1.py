from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Function to perform object detection using YOLO
def detect_memory(image_path):
    # Load YOLO model and weights
    net = cv2.dnn.readNet("yolo_model.weights", "yolo_model.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Load image
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process YOLO outputs
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust confidence threshold as needed
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to remove redundant boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    memory_boxes = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            # Draw bounding box around memory component
            x, y, w, h = boxes[i]
            memory_boxes.append({"x": x, "y": y, "width": w, "height": h})

    return memory_boxes

@app.route('/detect_memory', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save uploaded image
    image_path = "uploaded_image.jpg"
    file.save(image_path)

    # Perform memory detection
    memory_boxes = detect_memory(image_path)

    # Return detected memory boxes
    return jsonify({'memory_boxes': memory_boxes})

if __name__ == '__main__':
    app.run(debug=True)
