import numpy as np
import cv2


class Pupil(object):
    # 이 클래스는 눈동자를 감지하고 눈동자의 위치를 추정
    
    def __init__(self, eye_frame, threshold):
        self.iris_frame = None # 눈동자 프레임 초기화
        self.threshold = threshold # 이진화 임계값 초기화
        self.x = None # 눈동자 X좌표 초기화
        self.y = None # 눈동자 Y좌표 초기화

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): 눈만 포함하고 있는 프레임
            threshold (int): 눈 프레임을 이진화하기 위한 임계값
        Returns:
            눈동자 하나만 포함하는 프레임 반환
        """
        kernel = np.ones((3, 3), np.uint8) # 모폴로지 연산을 위한 커널 생성
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15) # 양방향 필터링을 통해 노이즈 제거
        new_frame = cv2.erode(new_frame, kernel, iterations=3) # 침식 연산을 통해 눈동자 부분 강조
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1] # 이진화 수행

        return new_frame

    def detect_iris(self, eye_frame):
        """눈동자를 감지하고눈동자 위치 추정

        Arguments:
            eye_frame (numpy.ndarray): 눈만 포함하고 있는 프레임
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        # 눈동자 외곽선 검출

        contours = sorted(contours, key=cv2.contourArea)
        # 외곽선 크기순으로 정렬

        try:
            moments = cv2.moments(contours[-2]) # 가장 큰 두 번째 외곽선에서 모멘트 계산
            self.x = int(moments['m10'] / moments['m00']) # x좌표 게산
            self.y = int(moments['m01'] / moments['m00']) # y좌표 계산
        except (IndexError, ZeroDivisionError):
            pass
