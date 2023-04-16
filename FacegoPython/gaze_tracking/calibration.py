from __future__ import division
import cv2
from .pupil import Pupil # 같은 패키지 안에 있는 pupil 모듈 import


class Calibration(object):
    """
    이 클래스는 사람과 웹캠에 대해 최적의 이진화(threshold) 임계값을 찾아 
    동공 검출 알고리즘을 보정(calibrate)합니다.
    """
    
    def __init__(self):
        self.nb_frames = 140 # 이진화(threshold)값을 찾기 위해 분석할 프레임 수 
        self.thresholds_left = []  # 왼쪽 눈 동공 임계값 리스트
        self.thresholds_right = []  # 오른쪽 눈 동공 임계값 리스트

    def is_complete(self):
        """보정(calibration)이 완료되었는지 여부를 반환합니다."""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        """주어진 눈(왼쪽(0) or 오른쪽(1))에 대한 threshold값을 반환합니다."""
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        """눈의 표면에 차지하는 동공의 크기(비율)를 반환합니다.

        Argument:
            frame (numpy.ndarray): 이진화된 동공 프레임
        """
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        """눈의 표면에 차지하는 동공의 크기(비율)를 반환합니다.

        Argument:
            frame (numpy.ndarray): 이진화된 동공 프레임
        """
        average_iris_size = 0.48 # 인간의 눈동자가 차지하는 면적의 평균값
        trials = {} # 최적의 threshold 값을 찾기 위한 실험 결과를 담을 딕셔너리

        for threshold in range(5, 100, 5): #threshold 값을 5에서 100까지 5씩 증가시키면서 실험
            iris_frame = Pupil.image_processing(eye_frame, threshold) # Pupil 클래스에서 제공하는 이미지 처리 함수로 iris 프레임을 이진화
            trials[threshold] = Calibration.iris_size(iris_frame) # 이진화된 iris 프레임의 눈동자 면적을 계산하여 딕셔너리에 추가
        
        # trials 딕셔너리에서 average_iris_size에 가장 가까운 key-value 쌍 찾기
        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, side):
        """Improves calibration by taking into consideration the
        given image.

        Arguments:
            eye_frame (numpy.ndarray): Frame of the eye
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        threshold = self.find_best_threshold(eye_frame)# 현재 프레임의 최적 threshold 값 찾기

        if side == 0:# 왼쪽 눈인 경우
            self.thresholds_left.append(threshold) # 찾은 threshold 값 리스트에 추가
        elif side == 1:# 오른쪽 눈인 경우
            self.thresholds_right.append(threshold) # 찾은 threshold 값 리스트에 추가 
