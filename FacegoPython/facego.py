import cv2 #영상 처리
import mediapipe as mp # 머신러닝 프레임워크
import numpy as np # 다차원 배열 처리
from gaze_tracking import GazeTracking 
import serial
import time
import math

# 시리얼 통신 설정
# port = "/dev/ttyACM0"  # 아두이노 시리얼 포트에 따라 변경
# ser = serial.Serial(port, 9600, timeout=1)  # 아두이노와 9600 bps로 시리얼 통신, timeout은 1초로 설정

# GazeTracking 선언
gaze = GazeTracking()

# FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 비디오 캡쳐 객체 초기화
cap = cv2.VideoCapture(0)

# 웹 캠이 켜지지 않았다면
if not cap.isOpened():
    print("Could not open webcam")
    exit()

# 비디오 프레임을 캡처하고 처리하는 루프
while cap.isOpened():
    # 비디오에서 한 프레임을 캡처
    status, frame = cap.read()

    # 프레임 캡처에 실패한 경우
    if not status:
        print("Could not read frame")
        break

    # 프레임을 좌우 대칭으로 바꾸고, BGR에서 RGB로 색상 공간을 변환
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # 성능을 향상시키기 위해
    frame.flags.writeable = False

    # FaceMesh를 사용하여 얼굴 랜드마크를 가져옵니다
    results = face_mesh.process(frame)

    frame.flags.writeable = True

    # RGB에서 BGR로 색상 공간을 변환합니다
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 시선 정보 업데이트
    gaze.refresh(frame)
    # horizontal_ratio = gaze.horizontal_ratio()
    # vertical_ratio = gaze.vertical_ratio()
    
    if gaze.horizontal_ratio() is not None:
        horizontal_ratio = round(gaze.horizontal_ratio(), 4)
    else:
        horizontal_ratio = 0.0  # 예외 상황 처리

    if gaze.vertical_ratio() is not None:
        vertical_ratio = round(gaze.horizontal_ratio(), 4)
    else:
        vertical_ratio = 0.0  # 예외 상황 처리

    frame = gaze.annotated_frame() #업데이트된 프레임에 시선 정보 추가하여 보여줌
    text = ""
    
    if gaze.is_blinking(): # 눈 깜빡일 때
        text = "Eye Blinking"
    elif gaze.is_right(): # 오른쪽 볼 때
        text = "Eye Right"
    elif gaze.is_left(): # 왼쪽 볼 때 
        text = "Eye left"
    elif gaze.is_center(): # 정면 볼 때 
        text = "Eye center"
    left_pupil = gaze.pupil_left_coords() # 왼쪽 눈동자 좌표 가져오기
    right_pupil = gaze.pupil_right_coords() # 오른쪽 눈동자 좌표 가져오기

    cv2.putText(frame, text, (70, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2) # 프레임에 시선 정보 텍스트 추가
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (70, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) # 왼쪽 눈동자 좌표
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (70, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) # 오른쪽 눈동자 좌표
    cv2.putText(frame, "Horizontal Ratio: " + str(horizontal_ratio), (70, 195), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) # 시선의 수평 방향 0~1
    cv2.putText(frame, "Vertical Ratio: " + str(vertical_ratio), (70, 225), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) # 시선의 수직 방향 0~1

    # #RC카를 제어하는 코드
    # if horizontal_ratio < 0.5:
    #     # 좌회전
    #     ser.write(b'1')
    # elif horizontal_ratio > 0.7:
    #     # 우회전
    #     ser.write(b'2')
    # else:
    #     # 직진
    #     ser.write(b'0')

    # time.sleep(0.1)  # 100ms 대기 후 반복


    # Get the head pose using FaceMesh
    img_h, img_w, img_c = frame.shape # 이미지 높이, 너비, 채널
    face_3d = [] # 3D 좌표 리스트
    face_2d = [] # 2D 좌표 리스트

    if results.multi_face_landmarks: # 멀티 페이스 랜드마크가 있으면
        for face_landmarks in results.multi_face_landmarks: #각 페이스 랜드마크에 대해서
            for idx, lm in enumerate(face_landmarks.landmark): #랜드마크 인덱스와 좌표에 대해서
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1: # 코 좌표
                        nose_2d = (lm.x * img_w, lm.y * img_h) # 코 2D 좌표
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000) # 코 3D 좌표

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # 2D 좌표 가져오기
                    face_2d.append([x, y])

                    # 3D 좌표 가져오기
                    face_3d.append([x, y, lm.z])

            # NumPy 배열로 변환
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # 카메라 매트릭스
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # 거리 매트릭스
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # PnP 문제 풀기
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # 회전 행렬 얻기
            rmat, jac = cv2.Rodrigues(rot_vec)

            # 각도 가져오기
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Y축 회전 각도 가져오기
            x = angles[0] * 360
            y = angles[1] * 360

            # print(y)

            # 머리 기울기 확인
            if y < -10:
                text = "Head Left"
            elif y > 10:
                text = "Head Right"
            elif x < -10:
                text = "Head Down"
            else:
                text = "Head Forward"

            # 코 방향 표시
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            # 2D 이미지 상의 코 위치 좌표
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            # 3D 공간에서의 코 위치 좌표를 2D이미지상 좌표로 변환하여 좌표 추출
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            # 코 위치 좌표를 이용해 이미지 위에 빨간색 선 그리기
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # 이미지 위에 텍스트 추가
            cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # 결과 이미지를 윈도우 창에 표시
    cv2.imshow('Head Pose Estimation', frame)
    # q 눌러서 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()