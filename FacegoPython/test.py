import cv2
import numpy as np
import dlib
import pyautogui

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def calibrate_center():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            landmarks = predictor(gray, face)

            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        cv2.putText(frame, "Please look at the center of the screen", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Check if the user is looking at the center of the screen
        if 240 <= x <= 400 and 160 <= y <= 240:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
def calibrate_corners():
    corners = []
    messages = ["Please look at the top-left corner of the screen",
                "Please look at the top-right corner of the screen",
                "Please look at the bottom-right corner of the screen",
                "Please look at the bottom-left corner of the screen"]
    
    cap = cv2.VideoCapture(0)
    for message in messages:
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) > 0:
                face = faces[0]
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                landmarks = predictor(gray, face)

                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y

                cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Calibration", frame)
                cv2.waitKey(1)

                # Only use the coordinates of the first detected face
                corners.append((x, y))
                break

    cap.release()
    cv2.destroyAllWindows()

    return corners

def main():
    calibrate_center()
    corners = calibrate_corners()
    print("Corner coordinates:", corners)

if __name__ == "__main__":
    main()
