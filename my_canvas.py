from tkinter import *
from tkinter.simpledialog import askstring
from tkinter.messagebox import showinfo
from PIL import Image, ImageDraw
from numpy.ma.core import array


class MyCanvas:

    def __init__(self, recognize_func, retrain_func):
        self.__x = 0
        self.__y = 0

        self.__root = Tk()
        self.__root.title('Text recognizer')
        self.__root.resizable(False, False)

        self.__brush_size = 20
        self.__color = 'black'
        self.__answer = None

        self.__root.columnconfigure(3)
        self.__root.rowconfigure(2)

        self.__canvas = Canvas(self.__root, bg='white', width=504, height=504)
        self.__canvas.grid(row=1, column=0, columnspan=7, padx=5, pady=5)
        self.__canvas.bind('<B1-Motion>', self.__draw)

        self.__image = Image.new('1', (504, 504), 'white')
        self.__draw_img = ImageDraw.Draw(self.__image)

        self.__btn_clear = Button(self.__root, text='Clear', width=10, command=lambda: self.__clear_canvas())
        self.__btn_clear.grid(row=0, column=0)
        self.__btn_test = Button(self.__root, text='Test', width=10, command=lambda: self.__test(recognize_func))
        self.__btn_test.grid(row=0, column=1)
        self.__answer_lbl = Label(self.__root, text='Number is:', width=10)
        self.__answer_lbl.grid(row=0, column=2)
        self.__btn_wrong = Button(self.__root, text='Wrong', width=10, command=lambda: self.__retrain(retrain_func))
        self.__btn_wrong.grid(row=0, column=4)

    def __draw(self, event):
        x1, y1 = (event.x - self.__brush_size), (event.y - self.__brush_size)
        x2, y2 = (event.x + self.__brush_size), (event.y + self.__brush_size)
        self.__canvas.create_oval(x1, y1, x2, y2, fill=self.__color, width=0)
        self.__draw_img.ellipse((x1, y1, x2, y2), fill=self.__color, width=0)

    def __clear_canvas(self):
        self.__canvas.delete('all')
        self.__canvas['bg'] = 'white'
        self.__draw_img.rectangle((0, 0, 504, 504), width=0, fill='white')
        self.__answer_lbl.config(text=f"Number is:")
        self.__answer = None

    def __test(self, recognize_func):
        img = self.__image.resize((28, 28))
        img_array = array(img.getdata())
        has_no_data = True

        for pixel in img_array:
            if pixel < 255:
                has_no_data = False
                break

        if not has_no_data:
            self.__answer = recognize_func(img_array)
            self.__answer_lbl.config(text=f"Number is: {self.__answer}")

    def __retrain(self, retrain_func):
        if not (self.__answer is None):
            correct_answer = int(askstring(title='Wrong', prompt='What is the correct number?', parent=self.__root))
            if isinstance(correct_answer, int):
                showinfo(title='Training', message='Process training')

                img = self.__image.resize((28, 28))
                img_array = array(img.getdata())
                retrain_func(img_array, correct_answer)
                showinfo(title='Trained', message='Training is done')

    def show(self):
        self.__root.mainloop()
