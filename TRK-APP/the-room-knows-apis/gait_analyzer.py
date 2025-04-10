import cv2
import mediapipe as mp
import math
import os
import uuid

def compute_line_deviation(landmark1, landmark2, width, height):
    pt1 = [landmark1.x * width, landmark1.y * height]
    pt2 = [landmark2.x * width, landmark2.y * height]
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    raw_angle = abs(math.degrees(math.atan2(dy, dx)))
    adjusted_angle = min(raw_angle, 180 - raw_angle)
    return adjusted_angle

def analyze_gait(video_path, output_dir='uploads'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open video file {video_path}"}, None

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    visibility_threshold = 0.8
    measurement_threshold = 4.0
    abnormal_frame_ratio_threshold = 0.3

    required_indices = [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                        mp_pose.PoseLandmark.LEFT_HIP.value,
                        mp_pose.PoseLandmark.RIGHT_HIP.value]

    abnormal_count = 0
    total_count = 0

    output_filename = f"{uuid.uuid4()}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            joints_visible = all(landmarks[idx].visibility > visibility_threshold for idx in required_indices)

            if joints_visible:
                total_count += 1
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

                abnormal_flag = shoulder_tilt > measurement_threshold or hip_symmetry > measurement_threshold
                if abnormal_flag:
                    abnormal_count += 1
                    cv2.putText(image_rgb, "Abnormal Frame", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.putText(image_rgb, f"Shoulder Tilt: {shoulder_tilt:.2f} deg", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image_rgb, f"Hip Symmetry: {hip_symmetry:.2f} deg", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            else:
                cv2.putText(image_rgb, "Required joints missing", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            cv2.putText(image_rgb, "No landmarks detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if out_writer is None:
            out_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        out_writer.write(image_rgb)

    cap.release()
    if out_writer:
        out_writer.release()

    if total_count > 0:
        abnormal_ratio = abnormal_count / total_count
        result = {
            "total_frames": total_count,
            "abnormal_frames": abnormal_count,
            "abnormal_ratio": abnormal_ratio,
            "abnormal_gait": abnormal_ratio > abnormal_frame_ratio_threshold
        }
    else:
        result = {
            "error": "No high-confidence frames processed"
        }

    return result, output_path
