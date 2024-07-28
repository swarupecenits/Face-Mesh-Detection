import mediapipe as mp
import cv2 as cv

mp_drawing = mp.solutions.drawing_utils # help draw holistic models through opencv
mp_holistic = mp.solutions.holistic

cap=cv.VideoCapture(0)

# Initialize the holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret,frame=cap.read()

        # Recolor the feed
        image=cv.cvtColor(frame,cv.COLOR_BGR2RGB)

        # Make detections
        results=holistic.process(image)
        print(results.face_landmarks)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Draw the face landmarks
        mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACE_CONNECTIONS)
        
        
        cv.imshow('Raw Webcam Feed', image)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()

