import cv2
import time
import numpy as np
from datetime import datetime
import os
from tensorflow.keras.models import load_model

# Load the TensorFlow model
model = load_model("my_model.keras")

# Preprocess image for TensorFlow
def preprocess_image(image, target_size=(48, 48)):
    resized_image = cv2.resize(image, target_size)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    normalized_image = gray_image / 255.0
    return np.expand_dims(normalized_image, axis=(0, -1))

# Classify emotion using TensorFlow model
def classify_emotion(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)[0]
    emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    
    # Get the top three emotions and their probabilities
    top_indices = predictions.argsort()[-3:][::-1]
    top_emotions = [(emotion_labels[idx], predictions[idx]) for idx in top_indices]
    
    return top_emotions

# Capture images and classify emotions
camera = cv2.VideoCapture(1)

def main():
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Starting image collection at {cur_time}")
    
    if not camera.isOpened():
        print("Error: Could not access the camera.")
        exit()
    
    if input("Enter 'start' to begin: ") == "start":
        os.makedirs(f"photos/{cur_time}", exist_ok=True)
        with open(f"results_{cur_time}.txt", "w") as f:
            f.write("Image Path | Emotion Rankings (Emotion: Probability)\n")
        print("Starting image collection")        
    else:
        return
    
    img_count = 0
    while True:
        ret, frame = camera.read()
        if ret:
            filename = f"photos/{cur_time}/image_{img_count}.jpg"
            cv2.imwrite(filename, frame)
            
            # Get the top three emotions
            top_emotions = classify_emotion(frame)
            
            # Log and display the results
            print(f"Image: {filename}")
            print("Top Emotions:")
            for emotion, prob in top_emotions:
                print(f"{emotion}: {prob:.2f}")
            
            # Save results to text file
            with open(f"results_{cur_time}.txt", "a") as f:
                emotion_text = ", ".join([f"{emotion}: {prob:.2f}" for emotion, prob in top_emotions])
                f.write(f"{filename} | {emotion_text}\n")
            
            img_count += 1
        else:
            print("Failed to capture image.")
        time.sleep(1)

if __name__ == "__main__":
    main()
    camera.release()
    cv2.destroyAllWindows()
