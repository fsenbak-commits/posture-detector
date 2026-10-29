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
    countdown_start = time.time()
    while time.time() - countdown_start < 3:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        seconds_left = 3 - int(time.time() - countdown_start)
        cv2.putText(frame, str(seconds_left), (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 6)
        cv2.imshow("Posture Detector - Calibration", frame)
        cv2.waitKey(1)
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
    recover_threshold = baseline_ratio * 0.88
    print("baseline ratio:", baseline_ratio)
    print("slouch threshold:", slouch_threshold)

    bad_posture_start_time = None
    last_alert_time = None
    is_slouching = False

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
            if ratio <slouch_threshold:
                is_slouching =  True
            elif ratio > recover_threshold:
                is_slouching = False
            if is_slouching == True:
                color = (0,0,255)
            else:
                color =(0,255,0)
            cv2.putText(frame, f"ratio: {ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"threshold: {slouch_threshold:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"recover: {recover_threshold:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if is_slouching and bad_posture_start_time is None:
                bad_posture_start_time = time.time()
            elif is_slouching:
                if time.time() - bad_posture_start_time > 5:
                    if last_alert_time is None or time.time() - last_alert_time > 60:
                        subprocess.run(["osascript", "-e", 'display notification "You have been slouching — sit up straight!" with title "Posture Check"'])
                        last_alert_time = time.time()
            else:
                bad_posture_start_time = None
                last_alert_time = None

                
            
        cv2.imshow("Posture Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()