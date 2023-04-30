import cv2
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import facego
import threading
from Arduino.socket import client
import TestClient

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # OpenCV 비디오 스트림 초기화
        self.video_stream = cv2.VideoCapture(0)

        # Tkinter 캔버스 초기화
        self.canvas = tk.Canvas(window, width=self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # PIL 이미지 초기화
        self.image = Image.open("face.png")
        self.image = self.image.resize((int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                         int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        self.delay = 15
        self.update()

        label = tk.Label(window, text=facego.text2)
        label.place(x=0, y=0)

        self.window.mainloop()

    def update(self):
        # 비디오 스트림에서 프레임 읽기
        ret, frame = self.video_stream.read()

        # 프레임과 PIL 이미지 합성
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_image = ImageTk.PhotoImage(pil_image)
        image_with_overlay = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0.7,
                                             cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR), 1.0, 0)

        # 합성된 이미지를 Tkinter 캔버스에 표시
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(image_with_overlay))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)



if __name__ == '__main__':
    App(tk.Tk(), "Tkinter and OpenCV")
    # t1 = threading.Thread(target=facego.test)
    # t1.start()
    # t2 = threading.Thread(target=client.test)
    # t2.start()
    t = threading.Thread(target=TestClient.test)
    t.start()