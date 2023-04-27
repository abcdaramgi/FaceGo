import facego
from tkinter import *
from PIL import ImageTk, Image

def main():
    tkinter()

def tkinter():
    root = Tk()
    root.title("tkinter_practice")
    root.geometry("640x640")

    # 이미지 불러오기
    image = Image.open("face.png")
    image = image.resize((int(image.size[0] / 2), int(image.size[1] / 2)))
    image = ImageTk.PhotoImage(image)

    # 레이블에 이미지 배치하기
    label = Label(root, image=image)
    label.pack()

    label2 = Label(root, text=facego.text)
    label2.place(x=10, y=10)

    root.mainloop()

if __name__ == "__main__":
    main()
