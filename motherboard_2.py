from flask import Flask, request, jsonify
import cv2
import os

app = Flask(__name__)

# Function to draw bounding boxes around memory components
def draw_boxes(image, annotations):
    for annotation in annotations:
        x = annotation['x']
        y = annotation['y']
        w = annotation['width']
        h = annotation['height']
        # Draw bounding box on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# Function to load YOLO annotations
def load_annotations(image_filename):
    annotation_filename = os.path.join("C:\Users\hp\Documents\Python\Python_Beginner\Data_Science\ds-task3\mem", image_filename[:-4] + ".txt")

    with open(annotation_filename, "r") as f:
        lines = f.readlines()
    annotations = []
    for line in lines:
        # Parse annotation line
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        # Convert YOLO format to (x, y, w, h) format
        x = int((x_center - width / 2) * image_width)
        y = int((y_center - height / 2) * image_height)
        w = int(width * image_width)
        h = int(height * image_height)
        annotations.append({"class_id": class_id, "x": x, "y": y, "width": w, "height": h})
    return annotations

@app.route('/detect_memory', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save uploaded image
    image_path = "C:\Users\hp\Documents\Python\Python_Beginner\Data_Science\ds-task3\mem"
    file.save(image_path)

    # Load image
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Load annotations
    annotations = load_annotations(os.path.basename(image_path))

    # Draw bounding boxes around memory components
    annotated_image = draw_boxes(image.copy(), annotations)

    # Return annotated image
    _, img_encoded = cv2.imencode('.jpg', annotated_image)
    response = img_encoded.tobytes()
    return Response(response=response, content_type='image/jpg')

if __name__ == '__main__':
    app.run(debug=True)
