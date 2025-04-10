import cv2
import mediapipe as mp
import math
import numpy as np
import time
import os
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
    video_path = r"C:\Users\rkt20\Downloads\Untitled video - Made with Clipchamp (1).mp4"  # Replace with your recorded video file path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties for VideoWriter setup.
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object.
    # You can change the codec if needed. For example, use "mp4v" for MP4 files.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (frame_width, frame_height))

    # Initialize MediaPipe Pose.
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Use a high confidence threshold.
    visibility_threshold = 0.8
    required_indices = [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                        mp_pose.PoseLandmark.LEFT_HIP.value,
                        mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Threshold for individual measurement.
    measurement_threshold = 4.0
    # Overall abnormal gait defined as abnormal in more than 30% of valid frames.
    abnormal_frame_ratio_threshold = 0.3

    abnormal_frame_count = 0
    total_frame_count = 0
    shoulder_tilts = []
    hip_symmetries = []

    missing_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height_frame, width_frame, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        joints_visible = False
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            joints_visible = all(landmarks[idx].visibility > visibility_threshold for idx in required_indices)
        else:
            joints_visible = False

        if not joints_visible:
            if missing_start_time is None:
                missing_start_time = time.time()
            else:
                elapsed = time.time() - missing_start_time
                if elapsed >= 2.0:
                    print("Required joints not detected with high confidence for 2 seconds. Exiting.")
                    break
            cv2.putText(image_rgb, "High confidence joints missing", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            missing_start_time = None
            total_frame_count += 1

            shoulder_tilt = compute_line_deviation(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                width_frame, height_frame
            )
            hip_symmetry = compute_line_deviation(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                width_frame, height_frame
            )
            shoulder_tilts.append(shoulder_tilt)
            hip_symmetries.append(hip_symmetry)

            if shoulder_tilt > measurement_threshold or hip_symmetry > measurement_threshold:
                abnormal_frame_count += 1
                cv2.putText(image_rgb, "Abnormal Frame", (50, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(image_rgb, f"Shoulder Tilt: {shoulder_tilt:.2f} deg", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image_rgb, f"Hip Symmetry: {hip_symmetry:.2f} deg", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Recorded Video - Gait Analysis", image_rgb)
        # Write the frame with annotations to the output file.
        out.write(image_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Save and close the output video.
    cv2.destroyAllWindows()

    if shoulder_tilts:
        avg_shoulder_tilt = sum(shoulder_tilts) / len(shoulder_tilts)
        print("Average Shoulder Tilt (deviation from horizontal): {:.2f} degrees".format(avg_shoulder_tilt))
    else:
        avg_shoulder_tilt = 0
        print("No shoulder tilt data recorded.")

    if hip_symmetries:
        avg_hip_symmetry = sum(hip_symmetries) / len(hip_symmetries)
        print("Average Hip Symmetry (deviation from horizontal): {:.2f} degrees".format(avg_hip_symmetry))
    else:
        avg_hip_symmetry = 0
        print("No hip symmetry data recorded.")

    if total_frame_count > 0:
        abnormal_ratio = abnormal_frame_count / total_frame_count
        #print(f"Total high-confidence frames: {total_frame_count}")
       # print(f"Abnormal frames: {abnormal_frame_count}")
        print(f"Abnormal frame ratio: {abnormal_ratio:.2f}")
        if abnormal_ratio > abnormal_frame_ratio_threshold:
            print("Abnormal gait detected")
        else:
            print("Gait appears normal")
    else:
        print("No high-confidence frames processed")
print(os.getcwd())
if __name__ == "__main__":
    main()
