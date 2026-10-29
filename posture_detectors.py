import cv2
import mediapipe as mp
import time
import subprocess


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def get_posture_ratio(results, frame):
    if not results.pose_landmarks:
        return None
    left_shoulder = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                     int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
    right_shoulder = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                      int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
    left_ear = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
    shoulder_distance = ((left_shoulder[0] - right_shoulder[0]) ** 2 + (left_shoulder[1] - right_shoulder[1]) ** 2) ** 0.5 #euclidean distance between shoulders
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    vertical_gap = shoulder_y - left_ear[1]
    ratio = vertical_gap / shoulder_distance
    return ratio
    
if __name__ == "__main__":
    print("Please maintain a good posture for 5 seconds to calibrate the system.")
    cap = cv2.VideoCapture(0)
    calibration_values = []
    callibration_start_time = time.time()
    

    while time.time() - callibration_start_time < 5:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        ratio = get_posture_ratio(results, frame)
        if ratio is not None:
            calibration_values.append(ratio)

        cv2.imshow("Posture Detector - Calibration", frame)
        cv2.waitKey(1)

    # moved out here — only checked ONCE, after all 5 seconds have passed
    if not calibration_values:
        print("No posture data detected. Please ensure your camera is positioned correctly and try again.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    baseline_ratio = sum(calibration_values) / len(calibration_values) 

    
    slouch_threshold = baseline_ratio * 0.8  # 20% less than baseline ratio
    print("baseline ratio:", baseline_ratio)
    print("slouch threshold:", slouch_threshold)

    bad_posture_start_time = None
    alert_triggered = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            ratio = get_posture_ratio(results, frame)
            if ratio < slouch_threshold and bad_posture_start_time is None:
                bad_posture_start_time = time.time()
            elif ratio < slouch_threshold and bad_posture_start_time is not None:
                if time.time() - bad_posture_start_time > 5 and not alert_triggered:
                    subprocess.run(["osascript", "-e", 'display notification "You have been slouching — sit up straight!" with title "Posture Check"'])
                    alert_triggered = True
            else:
                bad_posture_start_time = None
                alert_triggered = False

                
            
        cv2.imshow("Posture Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()