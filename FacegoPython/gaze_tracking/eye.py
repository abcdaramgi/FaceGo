import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41] # 왼쪽 눈 랜드마크
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47] # 오른쪽 눈 랜드마크

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """두 점 사이의 중간점 x,y 반환

        Arguments:
            p1 (dlib.point): 첫 번째 점
            p2 (dlib.point): 두 번째 점
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """눈을 분리해서 얼굴의 다른 부분이 없는 프레임을 가져옴 ???
        Arguments:
            frame (numpy.ndarray): 얼굴이 있는 프레임
            landmarks (dlib.full_object_detection): 얼굴 영역의 얼굴 랜드마크
            points (list): 눈의 포인트(68 Multi-PIE 랜드마크에서)

        """
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # 눈만 보이도록 마스크 적용
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # 눈 영역으로 크롭
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """눈이 감겨 있는지 여부를 나타내는 비율을 계산
        눈의 너비를 높이로 나눈 값

        Arguments:
            landmarks (dlib.full_object_detection): 얼굴 영역의 얼굴 랜드마크입니다.
        points (list): 눈의 점들 (68 Multi-PIE 랜드마크)

        Returns:
            계산된 비율 값
        """
        # 눈의 왼쪽, 오른쪽, 위, 아래 점을 추출
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        # 눈의 너비와 높이를 계산
        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))


        try:
            # 눈의 너비를 높이로 나눈 값을 계산
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """새로운 프레임에서 눈을 감지하고 분리하고 보정 데이터를 전송
        그리고 Pupil 객체를 초기화


        Arguments:
           original_frame (numpy.ndarray): 사용자가 전달한 프레임
            landmarks (dlib.full_object_detection): 얼굴 영역의 얼굴 랜드마크
            side: 왼쪽 눈 (0) 또는 오른쪽 눈 (1) 인지 나타내는 값입
            calibration (calibration.Calibration): 이진화 임계값을 관리
        """
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return
        
        # 눈 깜빡임 비율 계산
        self.blinking = self._blinking_ratio(landmarks, points)
        # 눈 분리
        self._isolate(original_frame, landmarks, points)

        # 보정이 완료되지 않았으면, 보정 값을 평가합니다.
        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        # 눈동자 객체를 생성합니다.
        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
