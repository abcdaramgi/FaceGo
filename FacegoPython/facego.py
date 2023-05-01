import cv2 #영상 처리
import cv2 as cv
import utils, math
import keyboard as keyboard
import mediapipe as mp # 머신러닝 프레임워크
import numpy as np # 다차원 배열 처리
import dlib
import threading
from concurrent.futures import ThreadPoolExecutor

from gaze_tracking import GazeTracking 
# import serial
import time
import math

from tkinter import *
from PIL import ImageTk, Image
# 시리얼 통신 설정
# port = "/dev/ttyACM0"  # 아두이노 시리얼 포트에 따라 변경
# ser = serial.Serial(port, 9600, timeout=1)  # 아두이노와 9600 bps로 시리얼 통신, timeout은 1초로 설정

text = ""
text2 = ""


FONTS = cv.FONT_HERSHEY_COMPLEX

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    return mesh_coord


def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

def eyesExtractor(img, right_eye_coords, left_eye_coords,right_iris_coords,left_iris_coords):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 이미지의 크기 정보 추출
    dim = gray.shape

    # 영역 추출을 위한 마스크 생성
    mask = np.zeros(dim, dtype=np.uint8)

    # 오른쪽 눈 좌표를 이용하여 마스크에 해당 영역을 흰색으로 채우기
    cv.fillPoly(mask, [np.array(right_iris_coords, dtype=np.int32)], 255)
    # 왼쪽 눈 좌표를 이용하여 마스크에 해당 영역을 흰색으로 채우기
    cv.fillPoly(mask, [np.array(left_iris_coords, dtype=np.int32)], 255)

    # 마스크를 이용하여 원본 이미지에서 눈 영역 추출
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # 마스크를 이용하여 눈 영역이 아닌 부분은 회색으로 채우기
    eyes[mask == 0] = 155

    # 오른쪽 눈 좌표 중 x 좌표의 최대값, 최소값
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]

    # 오른쪽 눈 좌표 중 y 좌표의 최대값, 최소값
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # 왼쪽 눈
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # 추출된 눈 이미지
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    return cropped_right, cropped_left


# 눈동자 위치 추정
def positionEstimator(cropped_eye):
    h, w = cropped_eye.shape

    cv.imshow("crop eye",cropped_eye)
    # 눈 이미지에 가우시안 필터 적용
    gaussain_blur = cv.GaussianBlur(cropped_eye, (5, 5), 0)

    #filtered_image = np.where(gaussain_blur > np.quantile(gaussain_blur, 0.2), gaussain_blur, 0)
    
    # 이진화
    _t,threshed_eye = cv.threshold(gaussain_blur, -1, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # -1을 기준으로 이진화 된 이미지를 반환
    # cv.THRESH_BINARY는 이진화 임계값보다 큰 값은 모두 255로(흰색),
    # 작은 값은 모두 0으로(검정색) 바꿈
    # cv.THRESH_OTSU는 임계값을 자동으로 결정


    # print("best threshold",_t)
    #kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    #eye_erode_bin = cv.erode(threshed_eye, kernel2, iterations=2)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    eye_bin = cv.dilate(threshed_eye, kernel, iterations=1)

    #cv.imshow("ga",gaussain_blur)
    #cv.imshow("filter",filtered_image)
    cv.imshow("th",threshed_eye)
    cv.imshow("bin",eye_bin)
    
    #cv.imshow("eye",eye_bin)

    # 눈 임지를 5등분하여 각 영역별 픽셀 값 수 계산
    piece = int(w / 5)

    not_center_piece = int(piece*2)
    right_piece = eye_bin[0:h, 0:not_center_piece]
    center_piece = eye_bin[0:h, not_center_piece : not_center_piece+piece]
    left_piece = eye_bin[0:h, not_center_piece+piece+piece:w]

    # 각 영역의 픽셀값 수 계산
    eye_position, color = pixelCounter(left_piece, center_piece, right_piece)

    return eye_position, color

# 각 영역별로 픽셀 값을 계산하여 눈동자 위치를 반환하는 함수
def pixelCounter(first_piece, second_piece, third_piece):
    left_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    right_part = np.sum(third_piece == 0)

    eye_parts = [left_part, center_part, right_part]
    
    # 픽셀 값이 가장 많은 영역의 인덱스를 구함
    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ''

    # 인덱스에 따라 눈동자 위치를 반환
    if max_index == 0:
        pos_eye = "RIGHT"
        color = [utils.BLACK, utils.GREEN]
    elif max_index == 1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index == 2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye = "Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


# 눈동자 위치 추정
def recalibrate(cropped_eye):
    h, w = cropped_eye.shape

    cv.imshow("crop eye",cropped_eye)
    # 눈 이미지에 가우시안 필터 적용
    #(3,3) : 가우시안 블러 커널의 크기, 세로방향3개 가로방향 3개의 픽셀을참조하여 블러링, 클수록 블러링 효과가 강해짐(이미지의 세부 정도도 손실)
    #세 번째 인자인 0 : 가우시안 필터 함수의 x축과 y축의 표준편차(sigma)를 자동으로 계산

    gaussain_blur = cv.GaussianBlur(cropped_eye, (5, 5), 0)

    #이진화
    _t,threshed_eye = cv.threshold(gaussain_blur, -1, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # 70을 기준으로 이진화 된 이미지를 반환
    # cv.THRESH_BINARY는 이진화 임계값보다 큰 값은 모두 255로(흰색), 작은 값은 모두 0으로(검정색) 바꿈
    # cv.THRESH_OTSU는 임계값을 자동으로 결정
    # print("best threshold",_t)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    eye_bin = cv.dilate(threshed_eye, kernel, iterations=1)

    cv.imshow("th",threshed_eye)
    cv.imshow("bin",eye_bin)
    # cv.imshow('frame', frame)

    # 눈 이미지를 5등분하여 각 영역별 픽셀 값 수 계산
    piece = int(w / 5)

    not_center_piece = int(piece*2)
    right_piece = eye_bin[0:h, 0:not_center_piece]
    center_piece = eye_bin[0:h, not_center_piece : not_center_piece+piece]
    left_piece = eye_bin[0:h, not_center_piece+piece+piece:w]

    right_part = np.sum(right_piece == 0) #픽셀 값
    center_part = np.sum(center_piece == 0)
    left_part = np.sum(left_piece == 0)

    return right_part, center_part, left_part

def test():
    global text
    global text2
    print("시작해요~")
    # GazeTracking 선언
    gaze = GazeTracking()

    # FaceMesh 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 비디오 캡쳐 객체 초기화
    cap = cv2.VideoCapture(0)

    # 원본 동영상 크기 정보
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("원본 동영상 너비(가로) : {}, 높이(세로) : {}".format(w, h))

    # 동영상 크기 변환
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 가로
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 세로

    # 변환된 동영상 크기 정보
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("변환된 동영상 너비(가로) : {}, 높이(세로) : {}".format(w, h))

    # 웹 캠이 켜지지 않았다면
    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

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

        # # 시선 정보 업데이트
        # gaze.refresh(frame)
        # # horizontal_ratio = gaze.horizontal_ratio()
        # # vertical_ratio = gaze.vertical_ratio()

        # if gaze.horizontal_ratio() is not None:
        #     horizontal_ratio = round(gaze.horizontal_ratio(), 4)
        # else:
        #     horizontal_ratio = 0.0  # 예외 상황 처리

        # if gaze.vertical_ratio() is not None:
        #     vertical_ratio = round(gaze.horizontal_ratio(), 4)
        # else:
        #     vertical_ratio = 0.0  # 예외 상황 처리

        # frame = gaze.annotated_frame() #업데이트된 프레임에 시선 정보 추가하여 보여줌
        # # text = ""
        # # text2 = ""

        # if gaze.is_blinking(): # 눈 깜빡일 때
        #     text = "Eye Blinking"
        # elif gaze.is_right(): # 오른쪽 볼 때
        #     text = "Eye Right"
        # elif gaze.is_left(): # 왼쪽 볼 때
        #     text = "Eye left"
        # elif gaze.is_center(): # 정면 볼 때
        #     text = "Eye center"
        # left_pupil = gaze.pupil_left_coords() # 왼쪽 눈동자 좌표 가져오기
        # right_pupil = gaze.pupil_right_coords() # 오른쪽 눈동자 좌표 가져오기

        # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2) # 프레임에 시선 정보 텍스트 추가
        # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) # 왼쪽 눈동자 좌표
        # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) # 오른쪽 눈동자 좌표
        # cv2.putText(frame, "Horizontal Ratio: " + str(horizontal_ratio), (90, 195), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) # 시선의 수평 방향 0~1
        # cv2.putText(frame, "Vertical Ratio: " + str(vertical_ratio), (90, 225), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) # 시선의 수직 방향 0~1
    


        # Get the head pose using FaceMeshssssssssss
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

                printx = x;
                printy = y;

                # print(y)

                # 머리 기울기 확인
                if y < -5:
                    text2 = "Head Left"
                    print(text2)
                elif y > 5:
                    text2 = "Head Right"
                    print(text2)
                elif x < -2:
                    text2 = "Head Down"
                    print(text2)
                else:
                    text2 = "Head Forward"
                    print(text2)

                # 코 방향 표시
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                # 2D 이미지 상의 코 위치 좌표
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                # 3D 공간에서의 코 위치 좌표를 2D이미지상 좌표로 변환하여 좌표 추출
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                # 코 위치 좌표를 이용해 이미지 위에 빨간색 선 그리기
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

                # 이미지 위에 텍스트 추가
                cv2.putText(frame, text2, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(printx, printy, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 결과 이미지를 윈도우 창에 표시
        # cv2.imshow('Head Pose, Gaze Tracking Estimation', frame)
        # q 눌러서 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break