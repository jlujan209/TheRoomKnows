import cv2
import time
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
from datetime import datetime
import os
from facenet_pytorch import MTCNN
from PIL import Image

#ds = load_dataset("dannyroxas/EMOTION_PHOTOS")
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
token = "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"
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
camera = cv2.VideoCapture(0)
face_detector = MTCNN(keep_all=False)

def detect_and_crop_single_face(image_path, output_path=None):
    image = Image.open(image_path).convert("RGB")
    
    # Detect faces
    boxes, _ = face_detector.detect(image)

    if boxes is None:
        print("No faces detected.")
        return None

    if len(boxes) > 1:
        print("Multiple faces detected.")
        return None

    # Only one face detected, crop it
    x1, y1, x2, y2 = map(int, boxes[0])
    face = image.crop((x1, y1, x2, y2))

    # Save the cropped face if an output path is provided
    if output_path:
        face.save(output_path)
        print(f"Cropped face saved to {output_path}")

    return face

def main():
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Starting image collection at {cur_time}")
    if not camera.isOpened():
        print("Error: Could not access the camera.")
        exit()
    if input("enter 'start' to begin: ") == "start":
        os.makedirs("csvs", exist_ok=True)
        with open(f"csvs/results_{cur_time}.csv", "w") as f:
            f.write("image path, output\n")
        print("starting image collection")        
    else:
        return 0
    img_count = 0
    os.makedirs(f"photos/{cur_time}")
    while True:
        ret, frame = camera.read()  # Capture a frame
        if ret:
            filename = f"photos/{cur_time}/image_{img_count}.jpg"
            cv2.imwrite(filename, frame)  # Save the captured image
            cropped_frame = detect_and_crop_single_face(filename, filename)
            if cropped_frame is None:
                print("either 0 or >1 faces detected")
                continue 
            emotion = classify_emotion(cropped_frame)
            print(f"emotion classified {emotion}")
            with open(f"results_{cur_time}.csv", "a") as f:
                f.write(f"{filename}, {emotion}\n")
            img_count += 1
        else:
            print("Failed to capture image.")
        time.sleep(1)

def classify_emotion(image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    print(logits)
    predicted_class_idx = logits.argmax(-1).item()

    predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class

if __name__ == "__main__":
    main()
    camera.release()
    cv2.destroyAllWindows()