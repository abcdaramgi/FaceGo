import tkinter as tk
from PIL import ImageTk, Image
import cv2

class VideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (False, None)

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)

        self.video_source = video_source
        self.vid = VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        self.btn_snapshot = tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def snapshot(self):
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite("snapshot.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def main():
    tkinter()

def tkinter():
    root = Tk()
    root.title("tkinter_practice")
    root.geometry("640x480")

    # 이미지 불러오기
    image = Image.open("FacegoPython/face.png")
    image = ImageTk.PhotoImage(image)

    # 레이블에 이미지 배치하기
    label = Label(root, image=image)
    label.pack()

    root.mainloop()

if __name__ == "__main__":
    #main()
    App(tk.Tk(), "Tkinter and OpenCV")
