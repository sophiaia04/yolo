import cv2
import argparse
import numpy as np

# Define the detect_objects function
def detect_objects(image_path):
    # Hardcoded paths for YOLO config, weights, and classes files
    config_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    classes_path = "yolov3.txt"
    image = cv2.imread(image_path)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights_path, config_path)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes, COLORS)

    return image

# Helper functions
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 4)

# Main execution
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help = 'path to input image')
    ap.add_argument('-c', '--config', required=True, help = 'path to yolo config file')
    ap.add_argument('-w', '--weights', required=True, help = 'path to yolo pre-trained weights')
    ap.add_argument('-cl', '--classes', required=True, help = 'path to text file containing class names')
    args = ap.parse_args()

    output_image = detect_objects(args.image, args.config, args.weights, args.classes)

    # Display and save the output image
    cv2.imshow("object detection", output_image)
    cv2.waitKey()
    cv2.imwrite("object-detection.jpg", output_image)
    cv2.destroyAllWindows()
