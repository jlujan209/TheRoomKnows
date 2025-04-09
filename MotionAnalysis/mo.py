import cv2
import mediapipe as mp
import math
import numpy as np
import time

def compute_line_deviation(landmark1, landmark2, width, height):
    """
    Computes the deviation from horizontal (in degrees) for the line connecting
    two landmarks.
    """
    pt1 = [landmark1.x * width, landmark1.y * height]
    pt2 = [landmark2.x * width, landmark2.y * height]
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    raw_angle = abs(math.degrees(math.atan2(dy, dx)))
    adjusted_angle = min(raw_angle, 180 - raw_angle)
    return adjusted_angle

def main():
    video_path = r"C:\Users\rkt20\Downloads\Untitled video - Made with Clipchamp.mp4"  # Replace with your recorded video file path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Use a high visibility threshold for high-confidence detections
    visibility_threshold = 0.8
    required_indices = [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                        mp_pose.PoseLandmark.LEFT_HIP.value,
                        mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Threshold for abnormal measurement (in degrees)
    measurement_threshold = 4.0
    # We'll consider the gait abnormal if more than this fraction of the frames are abnormal.
    abnormal_frame_ratio_threshold = 0.3

    abnormal_count = 0  # Count of frames flagged as abnormal
    total_count = 0     # Count of frames with high-confidence landmarks

    # Process each frame in the recorded video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Only proceed if landmarks are detected.
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Check if the required joints (shoulders and hips) are detected with high confidence.
            joints_visible = all(landmarks[idx].visibility > visibility_threshold for idx in required_indices)
            if joints_visible:
                total_count += 1
                # Compute shoulder tilt and hip symmetry.
                shoulder_tilt = compute_line_deviation(
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                    width, height
                )
                hip_symmetry = compute_line_deviation(
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                    width, height
                )
                # Frame is abnormal if either measurement is above the threshold.
                if shoulder_tilt > measurement_threshold or hip_symmetry > measurement_threshold:
                    abnormal_count += 1
                    abnormal_flag = True
                else:
                    abnormal_flag = False

                # Overlay current measurements on the frame.
                cv2.putText(image_rgb, f"Shoulder Tilt: {shoulder_tilt:.2f} deg", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image_rgb, f"Hip Symmetry: {hip_symmetry:.2f} deg", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                if abnormal_flag:
                    cv2.putText(image_rgb, "Abnormal Frame", (50, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(image_rgb, "Required joints missing", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            cv2.putText(image_rgb, "No landmarks detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Recorded Video - Gait Analysis", image_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Report the abnormal gait ratio over the video.
    if total_count > 0:
        abnormal_ratio = abnormal_count / total_count
        print(f"Total high-confidence frames: {total_count}")
        print(f"Abnormal frames: {abnormal_count}")
        print(f"Abnormal frame ratio: {abnormal_ratio:.2f}")
        if abnormal_ratio > abnormal_frame_ratio_threshold:
            print("Abnormal gait detected")
        else:
            print("Gait appears normal")
    else:
        print("No high-confidence frames processed")

if __name__ == "__main__":
    main()
