import cv2
import time
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
from datetime import datetime
import os

ds = load_dataset("dannyroxas/EMOTION_PHOTOS")
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
token = "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"
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
camera = cv2.VideoCapture(1)

def main():
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensor_output_file = f"tensot_results_{cur_time}.csv"
    print(f"Starting image collection at {cur_time}")
    if not camera.isOpened():
        print("Error: Could not access the camera.")
        exit()
    if input("enter 'start' to begin: ") == "start":
        with open(f"results_{cur_time}.csv", "w") as f:
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
            emotion = classify_emotion(frame, filename=tensor_output_file, img_filename=filename)
            print(f"emotion classified {emotion}")
            with open(f"results_{cur_time}.csv", "a") as f:
                f.write(f"{filename}, {emotion}\n")
            img_count += 1
        else:
            print("Failed to capture image.")
        time.sleep(1)

def classify_emotion(image, filename=None, img_filename=None):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    print(logits)
    if filename and img_filename:
        with open(filename, "a") as f:
            f.write(f'{img_filename}, "{str(logits)}"\n')
    predicted_class_idx = logits.argmax(-1).item()

    predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class

if __name__ == "__main__":
    main()
    camera.release()
    cv2.destroyAllWindows()