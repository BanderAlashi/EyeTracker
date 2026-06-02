import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Iris + eye landmarks
LEFT_IRIS = [474, 475, 476, 477]
LEFT_EYE_LEFT = 263
LEFT_EYE_RIGHT = 362

# State tracking
last_state = ""
look_start_time = None
look_confirmed = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape

    # Default values every frame (IMPORTANT)
    state = "NO_FACE"
    text = "No Face"
    color = (255, 255, 255)

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        # Draw face mesh (RED)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 0, 255),
                thickness=1,
                circle_radius=1
            )
        )

        # Convert landmarks
        mesh_points = np.array([
            np.multiply([p.x, p.y], [w, h]).astype(int)
            for p in face_landmarks.landmark
        ])

        # Iris center
        iris_center = mesh_points[LEFT_IRIS].mean(axis=0).astype(int)

        # Eye corners
        left_point = mesh_points[LEFT_EYE_LEFT]
        right_point = mesh_points[LEFT_EYE_RIGHT]

        eye_width = right_point[0] - left_point[0]

        # Prevent division error
        if eye_width != 0:

            iris_position = (iris_center[0] - left_point[0]) / eye_width

            # Gaze logic
            if 0.4 < iris_position < 0.6:
                text = "Looking At Camera"
                color = (0, 255, 0)
                state = "FRONT"

            elif iris_position <= 0.4:
                text = "Looking Right"
                color = (0, 0, 255)
                state = "LEFT"

            else:
                text = "Looking Left"
                color = (0, 0, 255)
                state = "RIGHT"

    # -----------------------------
    # 3 SECOND ATTENTION LOGIC
    # -----------------------------
    current_time = time.time()

    if state == "FRONT":

        if look_start_time is None:
            look_start_time = current_time

        duration = current_time - look_start_time

        if duration >= 3 and not look_confirmed:
            print("LOOK AT CAMERA")
            look_confirmed = True

    else:
        look_start_time = None
        look_confirmed = False

    # Print only when state changes
    if state != last_state:
        print(state)
        last_state = state

    # Show text
    cv2.putText(
        frame,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Face + Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()