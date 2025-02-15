import cv2
import mediapipe as mp
import pyautogui
import time

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
blink_start_time = None
click_threshold = 0.018
long_blink_threshold = 0.6
scroll_sensitivity = 50
last_blink_time = 0
smoothing_factor = 0.2
prev_screen_x, prev_screen_y = 0, 0

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y


                screen_x = (prev_screen_x * (1 - smoothing_factor)) + (screen_x * smoothing_factor)
                screen_y = (prev_screen_y * (1 - smoothing_factor)) + (screen_y * smoothing_factor)

                pyautogui.moveTo(screen_x, screen_y, duration=0.05)
                prev_screen_x, prev_screen_y = screen_x, screen_y

        left = [landmarks[145], landmarks[159]]
        left_y_values = [landmark.y for landmark in left]
        blink_diff = abs(left_y_values[0] - left_y_values[1])

        if blink_diff < click_threshold:
            if blink_start_time is None:
                blink_start_time = time.time()
        else:
            if blink_start_time is not None:
                blink_duration = time.time() - blink_start_time
                blink_start_time = None

                if blink_duration > long_blink_threshold:
                    pyautogui.rightClick()
                elif time.time() - last_blink_time < 0.3:
                    pyautogui.doubleClick()
                else:
                    pyautogui.click()
                    last_blink_time = time.time()

        gaze_y = landmarks[4].y
        if gaze_y < 0.4:
            pyautogui.scroll(scroll_sensitivity)
        elif gaze_y > 0.6:
            pyautogui.scroll(-scroll_sensitivity)

    cv2.imshow('EYE GESTURE CONTROL', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
