import cv2
import numpy as np
import os
from logger import setup_logger

# Setup logger
logger = setup_logger()

# Define paths

path = r"yolov3-coco"  # Ensure this path is correct
weight = os.path.join(path, "yolov3.weights")
cfg = os.path.join(path, "yolov3.cfg")
names_file = os.path.join(path, "coco.names")


# Debug prints to check paths
print("Weight file path:", weight)
print("Config file path:", cfg)
print("Names file path:", names_file)

logger.info("Loading YOLO model...")
net = cv2.dnn.readNetFromDarknet(cfg, weight)

logger.info("Loading class labels...")
with open(names_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten().tolist()]

def is_person_horizontal(box):
    x, y, w, h = box
    aspect_ratio = w / h
    return aspect_ratio > 1.2  # Example threshold to consider a person horizontal

def process_frame(frame, net, output_layers, classes):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            
            if is_person_horizontal(boxes[i]):
                label += " (Horizontal)"
                logger.info(f"Detected horizontal person: {label} at {x}, {y}, {w}, {h}")
            else:
                label += " (Vertical)"
                logger.info(f"Detected vertical person: {label} at {x}, {y}, {w}, {h}")
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    logger.info("Starting video capture...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame.")
            break

        frame = process_frame(frame, net, output_layers, classes)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Quitting the application.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Video capture stopped and windows closed.")

if __name__ == "__main__":
    main()