import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import threading
import socket
import schedule
from concurrent.futures import ThreadPoolExecutor
import cv2 #영상 처리

from FacegoPython.facego_eyetracker import color

eye_set_complete = False

text = ""
text2 = ""
eyemessage = ""

HOST = '192.168.137.4'
PORT = 9999
interval = 0.1

# # socket 객체 생성
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # 지정한 HOST와 PORT 사용하여 서버 접속
# client_socket.connect((HOST, PORT))

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

map_face_mesh = mp.solutions.face_mesh
camera = cv.VideoCapture(0)



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

    # cv.imshow("crop eye",cropped_eye)
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
    # cv.imshow("th",threshed_eye)
    # cv.imshow("bin",eye_bin)
    
    #cv.imshow("eye",eye_bin)

    # 눈 임지를 5등분하여 각 영역별 픽셀 값 수 계산
    piece = int(w / 5)

    not_center_piece = int(piece*2)
    right_piece = eye_bin[0:h, 0:not_center_piece]
    center_piece = eye_bin[0:h, not_center_piece : not_center_piece+piece]
    left_piece = eye_bin[0:h, not_center_piece+piece+piece:w]

    # 각 영역의 픽셀값 수 계산
    eye_position, color, eyemessage = pixelCounter(left_piece, center_piece, right_piece)

    return eye_position, color, eyemessage

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
        print(pos_eye)
        eyemessage = pos_eye
        if text2 == "Head Left":
            eyemessage = 'l'
        elif text2 == "Head Right":
            eyemessage = 'r'
        else :
            eyemessage = 's'
        color = [utils.BLACK, utils.GREEN]
    elif max_index == 1:
        pos_eye = 'CENTER'
        print(pos_eye)
        eyemessage = pos_eye
        if text2 == "Head Left":
            eyemessage = 'l'
        elif text2 == "Head Right":
            eyemessage = 'r'
        else :
            eyemessage = 'g'
        color = [utils.YELLOW, utils.PINK]
    elif max_index == 2:
        pos_eye = 'LEFT'
        print(pos_eye)
        eyemessage = pos_eye
        if text2 == "Head Left":
            eyemessage = 'l'
        elif text2 == "Head Right":
            eyemessage = 'r'
        else :
            eyemessage = 's'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye = "Closed"
        print(pos_eye)
        eyemessage = pos_eye
        if text2 == "Head Left":
            eyemessage = 'l'
        elif text2 == "Head Right":
            eyemessage = 'r'
        else :
            eyemessage = 's'
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color, eyemessage


# 눈동자 위치 추정
def recalibrate(cropped_eye):
    h, w = cropped_eye.shape

    # cv.imshow("crop eye",cropped_eye)
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

    # cv.imshow("th",threshed_eye)
    # cv.imshow("bin",eye_bin)
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



class Countdown(threading.Thread):
    def __init__(self, event, set_frame_callback):
        super().__init__()
        self.event = event
        self.is_countdown_finished = False
        self.countdown = 5
        self.set_frame_callback = set_frame_callback
        
    def run(self):
        while self.countdown > 0 and not self.event.is_set():
            print("CountDown run!")
            # 카운트다운 텍스트 생성
            text = f'{self.countdown}'
            (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 2, 2)
            x = int((640 - text_width) / 2)
            y = int((480 - text_height) / 2)
            frame = self.set_frame_callback()
            cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv.LINE_AA)
            
            # 1초 대기
            time.sleep(1)  
            self.countdown -= 1
        
        # 카운트다운 종료
        self.is_countdown_finished = True
        self.event.set()
        
def set_eyetracking(frame):
    count = 0
    event = threading.Event()
    countdown_thread = Countdown(event, lambda: frame)
    
    while not event.is_set():
        print("set_Eyetracking run~\n")
        # 좌측 동그라미 그리기
        cv.circle(frame, (240, 350), 30, (0, 0, 255), 2)

        event.wait()  # event가 set() 될 때까지 대기
        
        if countdown_thread.is_countdown_finished:
            # 카운트다운이 끝난 경우
            count += 1

        if count == 1:
            cv.circle(frame, (240, 350), 30, (0, 0, 255), -1) # 좌측 지우고
            cv.circle(frame, (680, 350), 30, (0, 0, 255), 2) # 우측 동그랑미 그리기
            countdown_thread.start()

        elif count == 2:
            utils.colorBackgroundText(frame, f'EyeTracking Setting Finish!', FONTS, 1.0, (600, 120), 2, color[0], color[1], 8, 8)
            break
        
       
    countdown_thread.join()
    cv.destroyAllWindows()

def real_set_eyetracking(frame, crop_left, crop_right):
    print("real_set_eyetracking run!")
    # while True:
    #     ret, frame = camera.read()
    #     if not ret:
    #         break

    #     # 좌우 반전
    #     frame = cv.flip(frame, 1)

    #     #이미지 크기 조정
    #     frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)

    #     #프레임 높이, 너비 추출
    #     frame_height, frame_width = frame.shape[:2]

    #     #RGB 프레임으로 변환
    #     rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    #     # FaceMesh 모듈의 process() 함수를 사용하여 얼굴 랜드마크 추출
    #     results = face_mesh.process(rgb_frame)

    #     if results.multi_face_landmarks:
    #         # 얼굴 랜드마크 좌표를 landmarksDetection() 함수를 사용하여 추출
    #         mesh_coords = landmarksDetection(frame, results, False)
            
    #         # 왼쪽 눈, 오른쪽 눈의 좌표를 사용하여 눈 주변의 다각형을 그리기
    #         cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
    #         cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
    #         cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
    #         cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            
    #         # 오른쪽 눈, 왼쪽 눈 이미지를 추출하여 eye_position, Estimator() 함수를 사용하여 눈의 위치 추정
    #         right_coords = [mesh_coords[p] for p in RIGHT_EYE]
    #         left_coords = [mesh_coords[p] for p in LEFT_EYE]
    #         right_iris_coords = [mesh_coords[p] for p in RIGHT_IRIS]
    #         left_iris_coords = [mesh_coords[p] for p in LEFT_IRIS]
    #         crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords,right_iris_coords,left_iris_coords)
            
    #         # cv.imshow('right', crop_right)
    #         # cv.imshow('left', crop_left)
            
    #         eye_position_right, color = positionEstimator(crop_right)
    #         utils.colorBackgroundText(frame, f'R: {eye_position_right}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)

    #         # 프레임에 눈의 위치와 색상을 나타내는 텍스트 추가
    #         eye_position_left, color = positionEstimator(crop_left)
    #         utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)
            
    #         current_r_pixel = recalibrate(crop_left)
    #         current_l_pixel = recalibrate(crop_right)
            
    #         utils.colorBackgroundText(frame, f'L: {current_l_pixel}', FONTS, 1.0, (350, 50), 2, color[0], color[1], 8, 8)
    #         utils.colorBackgroundText(frame, f'R: {current_r_pixel}', FONTS, 1.0, (600, 50), 2, color[0], color[1], 8, 8)
    count = 0
    cv.circle(frame, (240, 350), 30, (0, 0, 255), 2)
    utils.colorBackgroundText(frame, f'Look Left Circle', FONTS, 1.0, (350, 150), 2, color[0], color[1], 8, 8)

    countdown_thread = threading.Thread(Countdown())
    countdown_thread.start()
    utils.colorBackgroundText(frame, f'3', FONTS, 1.0, (350, 150), 2, color[0], color[1], 8, 8)

    utils.colorBackgroundText(frame, f'2', FONTS, 1.0, (350, 150), 2, color[0], color[1], 8, 8)

    utils.colorBackgroundText(frame, f'1', FONTS, 1.0, (350, 150), 2, color[0], color[1], 8, 8)
    
    if countdown_thread.is_countdown_finished:
        count += 1
        if count == 1: 
            lefteye_left_standard = recalibrate(crop_left)
            righteye_left_standard = recalibrate(crop_right)
            countdown_thread.start()

        elif count ==2:
            cv.circle(frame, (680, 350), 30, (0, 0, 255), 2)
            utils.colorBackgroundText(frame, f'Look Right Circle', FONTS, 1.0, (350, 150), 2, color[0], color[1], 8, 8)
            utils.colorBackgroundText(frame, f'3', FONTS, 1.0, (350, 150), 2, color[0], color[1], 8, 8)

            utils.colorBackgroundText(frame, f'2', FONTS, 1.0, (350, 150), 2, color[0], color[1], 8, 8)

            utils.colorBackgroundText(frame, f'1', FONTS, 1.0, (350, 150), 2, color[0], color[1], 8, 8)

            lefteye_right_standard = recalibrate(crop_left)
            righteye_right_standard = recalibrate(crop_right)

            utils.colorBackgroundText(frame, f'Set Finish!', FONTS, 1.0, (350, 150), 2, color[0], color[1], 8, 8)

            return lefteye_left_standard, righteye_left_standard, lefteye_right_standard, righteye_right_standard

def kokakola():
    print("시작해요~")
    global text2
    # 아이트래킹 프로그램 시작
    with map_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        
        set_finish = False

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            # 좌우 반전
            frame = cv.flip(frame, 1)

            #이미지 크기 조정
            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)

            #프레임 높이, 너비 추출
            frame_height, frame_width = frame.shape[:2]

            #RGB 프레임으로 변환
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            # FaceMesh 모듈의 process() 함수를 사용하여 얼굴 랜드마크 추출
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # 얼굴 랜드마크 좌표를 landmarksDetection() 함수를 사용하여 추출
                mesh_coords = landmarksDetection(frame, results, False)
                
                # 왼쪽 눈, 오른쪽 눈의 좌표를 사용하여 눈 주변의 다각형을 그리기
                cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                
                # 오른쪽 눈, 왼쪽 눈 이미지를 추출하여 eye_position, Estimator() 함수를 사용하여 눈의 위치 추정
                right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                left_coords = [mesh_coords[p] for p in LEFT_EYE]
                right_iris_coords = [mesh_coords[p] for p in RIGHT_IRIS]
                left_iris_coords = [mesh_coords[p] for p in LEFT_IRIS]
                crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords,right_iris_coords,left_iris_coords)
                
                # cv.imshow('right', crop_right)
                # cv.imshow('left', crop_left)
                
                eye_position_right, color, eyemessage = positionEstimator(crop_right)
                utils.colorBackgroundText(frame, f'R: {eye_position_right}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)

                # 프레임에 눈의 위치와 색상을 나타내는 텍스트 추가
                eye_position_left, color, eyemessage = positionEstimator(crop_left)
                utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)
                
                # if set_finish == False:
                #     eyetracking_standard = real_set_eyetracking(frame, crop_left, crop_right)
                #     set_finish = True

                # elif set_finish == True:
                current_r_pixel = recalibrate(crop_left)
                current_l_pixel = recalibrate(crop_right)
                utils.colorBackgroundText(frame, f'L: {current_l_pixel}', FONTS, 1.0, (350, 50), 2, color[0], color[1], 8, 8)
                utils.colorBackgroundText(frame, f'R: {current_r_pixel}', FONTS, 1.0, (600, 50), 2, color[0], color[1], 8, 8)

                # utils.colorBackgroundText(frame, f'LE.left_circle: {eyetracking_standard[0]}', FONTS, 1.0, (400, 450), 2, color[0], color[1], 8, 8)
                # utils.colorBackgroundText(frame, f'RE.left_circle: {eyetracking_standard[1]}', FONTS, 1.0, (400, 500), 2, color[0], color[1], 8, 8)
                # utils.colorBackgroundText(frame, f'LE.Right_circle: {eyetracking_standard[2]}', FONTS, 1.0, (400, 550), 2, color[0], color[1], 8, 8)
                # utils.colorBackgroundText(frame, f'RE.Right_circle: {eyetracking_standard[3]}', FONTS, 1.0, (400, 600), 2, color[0], color[1], 8, 8)
                
                # # ---------------------------------------------------------------------------------------------------------------------문제의 쓰레드 파트---------------------------
                # # ThreadPoolExecutor 객체 생성 executor가 계속해서 무한 생성 되서 500개 넘개 생김 이렇게 쓰면 안됨
                # executor = ThreadPoolExecutor(max_workers=1) # 최대 1개의 worker를 가질 수 있도록 설정
                
                # # set_complete가 FALSE이면, 아이트래킹 설정을 안했다면
                # if eye_set_complete == False:

                #     # threading.Thread(target=set_eyetracking()) 
                #     # submit() : 비동기 실행 결과를 나타내는 Future객체를 반환, 
                #     future = executor.submit(set_eyetracking, frame)
                    
                # # 아이트래킹 설정을 완료했고, future객체가 작업을 완료 하였으면
                # if eye_set_complete and future.done():

                #     result = future.result() # future.result : Future 객체의 실핼 결과를 가져옴
                #     utils.colorBackgroundText(frame, f'RE_LS: {result[0]}', FONTS, 1.0, (600, 120), 2, color[0], color[1], 8, 8)
                #     utils.colorBackgroundText(frame, f'RE_RS: {result[1]}', FONTS, 1.0, (600, 240), 2, color[0], color[1], 8, 8)
                #     utils.colorBackgroundText(frame, f'RE_LS: {result[2]}', FONTS, 1.0, (600, 360), 2, color[0], color[1], 8, 8)
                #     utils.colorBackgroundText(frame, f'RE_RS: {result[3]}', FONTS, 1.0, (600, 480), 2, color[0], color[1], 8, 8)
                # # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
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
                        headmessage = 'l'
                    elif y > 5:
                        text2 = "Head Right"
                        print(text2)
                        headmessage = 'r'
                    elif x < -2:
                        text2 = "Head Down"
                        print(text2)
                        headmessage = 's'
                    else:
                        text2 = "Head Forward"
                        print(text2)
                        headmessage = 'g'

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

                    # message = say_hello() # 함수 호출하여 결과를 변수에 저장
                    message = eyemessage
                    print(message) # 결과 출력(확인용)
                    time.sleep(interval) # 일정 시간 동안 대기
                    # # q 입력 시 종료
                    # if message == 'q':
                    #     client_socket.close()
                    # if message != None:
                    #     # 입력한 message 전송
                    #     client_socket.sendall(message.encode())

                    #     # 메시지 수신
                    #     data = client_socket.recv(1024)
                    #     print('Received', repr(data.decode()))
                    #     message = None
            # cv.imshow('frame', frame)
            key = cv.waitKey(2)
            if key == ord('q') or key == ord('Q'):
                break
        cv.destroyAllWindows()
        camera.release()


kokakola()