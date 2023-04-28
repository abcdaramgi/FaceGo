#
# # import facego
# from PIL import ImageTk, Image
# from tkinter import *
# import tkinter as tk
# from PIL import ImageTk, Image
# import cv2
#
# class VideoCapture:
#     def __init__(self, video_source=0):
#         self.vid = cv2.VideoCapture(video_source)
#         if not self.vid.isOpened():
#             raise ValueError("Unable to open video source", video_source)
#
#         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
#
#     def __del__(self):
#         if self.vid.isOpened():
#             self.vid.release()
#
#     def get_frame(self):
#         if self.vid.isOpened():
#             ret, frame = self.vid.read()
#             if ret:
#                 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             else:
#                 return (ret, None)
#         else:
#             return (False, None)
#
# class App:
#     def __init__(self, window, window_title, video_source=0):
#         self.window = window
#         self.window.title(window_title)
#
#         self.video_source = video_source
#         self.vid = VideoCapture(self.video_source)
#
#         self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
#         self.canvas.pack()
#
#         self.img = Image.open("face.png")
#         self.img = self.img.resize((200,200), Image.ANTIALIAS)
#         self.photo = ImageTk.PhotoImage(self.img)
#
#         self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
#         self.window.after(0, self.video_loop)
#
#         self.btn_snapshot = tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
#         self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
#
#         self.delay = 15
#         self.update()
#
#         self.window.mainloop()
#
#     def video_loop(self):
#         # Get frame from the video source
#         ret, frame = self.vid.get_frame()
#
#         if ret:
#             # Resize frame to fit the canvas
#             frame = cv2.resize(frame, (self.vid.width, self.vid.height))
#
#             # Convert the frame to RGB color space
#             cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # Convert the frame to PIL image format
#             self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
#
#             # Put the PIL image on the canvas
#             self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
#
#         # Repeat the self.video_loop function after 30 milliseconds
#         self.window.after(30, self.video_loop)
#     def update(self):
#         ret, frame = self.vid.get_frame()
#         if ret:
#             self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
#             self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
#
#         self.window.after(self.delay, self.update)
#
#     def snapshot(self):
#         ret, frame = self.vid.get_frame()
#         if ret:
#             cv2.imwrite("snapshot.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#
# def main():
#     tkinter()
#
# def tkinter():
#     root = Tk()
#     root.title("tkinter_practice")
#     root.geometry("640x480")
#
#     # 이미지 불러오기
#     image = Image.open("face.png")
#     image = ImageTk.PhotoImage(image)
#
#     # 레이블에 이미지 배치하기
#     label = Label(root, image=image)
#     label.pack()
#
#     root.mainloop()
#
# if __name__ == "__main__":
#     # main()
#     App(tk.Tk(), "Tkinter and OpenCV")
#     # tkinter()
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import facego
import threading
from Arduino.socket.client import test2



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
    t1 = threading.Thread(target=facego.test)
    t1.start()
    t2 = threading.Thread(target=test2())
    t2.start()