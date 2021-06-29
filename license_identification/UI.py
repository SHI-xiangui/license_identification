import cv2
import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow import keras
from core import *
from CNN import *
from core import *
from core import unet_predict, locate_and_correct
from show_Fun import *
from multiprocessing import Process, Queue


class Window:
    def __init__(self, win,ww,hh):
        self.win = win
        self.win.geometry("%dx%d+%d+%d" % (ww, hh, 200, 50))  # 界面启动时的初始位置
        self.win.title("车牌识别系统--xiaole")
        self.img_src_path = None
        self.gray_pic_path = None
        self.gray_histogram_pic = None
        edge_detect_pic = None

        self.unet = keras.models.load_model('unet.h5')
        self.cnn = keras.models.load_model('cnn.h5')
        print('正在启动中,请稍等...')
        cnn_predict(self.cnn, [np.zeros((80, 240, 3))])
        print("已启动,开始识别吧！")

        self.label_src = Label(self.win, text='欢迎使用车牌识别系统', font=('微软雅黑', 22), bg='red').place(x=10, y=10)

        # 创建一个图像预处理容器
        self.monty = LabelFrame(self.win, text="图像预处理")
        self.monty.place(relx=0.01, rely=0.12, relwidth=0.3, relheight=0.2)
        # # 创建子容器里的按钮1
        self.buttonx1 = Button(self.monty, text="获取图片", font=("宋体", 12), fg="red", width=10, height=1,
                               command=self.load_show_img)
        self.buttonx1.place(x=30, y=10)
        self.buttonx2 = Button(self.monty, text="灰度处理", font=("宋体", 12), fg="red", width=10, height=1,
                               command=self.gray_cope)
        self.buttonx2.place(x=165, y=10)
        # self.buttonx3 = Button(self.monty, text="直方图", font=("宋体", 12), fg="red", width=10, height=1, )
        # self.buttonx3.place(x=30, y=50)
        self.buttonx4 = Button(self.monty, text="边缘检测", font=("宋体", 12), fg="red", width=10, height=1,
                               command=self.edge_detect)
        self.buttonx4.place(x=100, y=60)

        # 创建车牌定位识别容器
        self.monty1 = LabelFrame(self.win, text="车牌定位")
        self.monty1.place(relx=0.01, rely=0.35, relwidth=0.3, relheight=0.13)
        # 创建子容器里的按钮1
        self.buttonx5 = Button(self.monty1, text="二值化", font=("宋体", 12), fg="red", width=10, height=1,
                               command=self.two_value)
        self.buttonx5.place(x=30, y=10)
        self.buttonx6 = Button(self.monty1, text="车牌矫正", font=("宋体", 12), fg="red", width=10, height=1,
                               command=self.lic_position)
        self.buttonx6.place(x=160, y=10)

        # 车牌识别功能按钮
        self.monty2 = LabelFrame(self.win, text="车牌识别")
        self.monty2.place(relx=0.01, rely=0.5, relwidth=0.3, relheight=0.46)
        # 创建子容器里的按钮1
        self.buttonx7 = Button(self.monty2, text="车牌识别", font=("宋体", 12), fg="black", bg="orange", width=10, height=1,
                               command=self.lic_display)
        self.buttonx7.place(x=100, y=30)
        self.buttonx8 = Button(self.monty2, text="一键清空", font=("宋体", 12), fg="black", bg="green", width=10, height=1,
                               command=self.clear)
        self.buttonx8.place(x=100, y=100)
        self.buttonx9 = Button(self.monty2, text="退出系统", font=("宋体", 12), fg="black", bg="red", width=10, height=1,
                               command=self.closeEvent)
        self.buttonx9.place(x=100, y=170)

        self.buttonx10 = Button(self.monty2, text="摄像头识别", font=("宋体", 12), fg="black", bg="red", width=10, height=1, command=self.video_stream)
        self.buttonx10.place(x=100, y=210)

        # 预处理图像
        self.monty3 = LabelFrame(self.win, text="车牌识别")
        self.monty3.place(relx=0.35, rely=0.03, relwidth=0.64, relheight=0.4)

        # 定位与识别图像
        self.monty3 = LabelFrame(self.win, text="定位与识别")
        self.monty3.place(relx=0.35, rely=0.48, relwidth=0.64, relheight=0.48)

        # 原图
        self.label_src1 = Label(self.win, text='原图', font=('微软雅黑', 13)).place(x=430, y=50)
        self.can_src1 = Canvas(self.win, width=190, height=130, bg='white', relief='solid', borderwidth=1)  # 原图画
        self.can_src1.place(x=360, y=90)

        # 灰度处理
        self.label_src2 = Label(self.win, text='灰度处理', font=('微软雅黑', 13)).place(x=630, y=50)
        self.can_src2 = Canvas(self.win, width=190, height=130, bg='white', relief='solid', borderwidth=1)  # 原图画
        self.can_src2.place(x=572, y=90)

        # 直方图处理
        # self.label_src3 = Label(self.win, text='直方图', font=('微软雅黑', 13)).place(x=714, y=50)
        # self.can_src3 = Canvas(self.win, width=130, height=130, bg='yellow', relief='solid', borderwidth=1)
        # self.can_src3.place(x=680, y=90)

        # 边缘检测
        self.label_src4 = Label(self.win, text='边缘检测', font=('微软雅黑', 13)).place(x=845, y=50)
        self.can_src4 = Canvas(self.win, width=190, height=130, bg='white', relief='solid', borderwidth=1)
        self.can_src4.place(x=784, y=90)

        # 二值化
        self.label_src5 = Label(self.win, text='二值化:', font=('微软雅黑', 13)).place(x=660, y=325)
        self.can_src5 = Canvas(self.win, width=220, height=60, bg='white', relief='solid', borderwidth=1)
        self.can_src5.place(x=750, y=305)

        # 车牌定位及矫正
        self.label_src6 = Label(self.win, text='车牌定位:', font=('微软雅黑', 13)).place(x=660, y=420)
        self.can_src6 = Canvas(self.win, width=220, height=60, bg='white', relief='solid', borderwidth=1)
        self.can_src6.place(x=750, y=402)

        # 识别车牌
        self.label_src7 = Label(self.win, text='识别车牌:', font=('微软雅黑', 13)).place(x=660, y=520)
        self.can_src7 = Canvas(self.win, width=220, height=60, bg='white', relief='solid', borderwidth=1)
        self.can_src7.place(x=750, y=500)

        # 摄像头识别
        # self.label_src7 = Label(self.win, text='识别车牌', font=('微软雅黑', 13)).place(x=830, y=445)
        # self.can_src8 = Canvas(self.win, width=250, height=240, bg='yellow', relief='solid', borderwidth=1)
        # self.can_src8.place(x=360, y=315)
        self.image_frame = Frame(master=self.win, width=220, height=220, bg='white', relief='solid',
                                    bd=1)  # label for the video frame
        self.image_frame.place(x=360, y=315)

    # 图片加载
    def load_show_img(self):
        self.clear()
        sv = StringVar()
        sv.set(askopenfilename())
        self.img_src_path = Entry(self.win, state='readonly', text=sv).get()  # 获取到所打开的图片
        img_open = Image.open(self.img_src_path)
        if img_open.size[0] * img_open.size[1] > 240 * 80:
            img_open = img_open.resize((190, 130), Image.ANTIALIAS)
            width = img_open.size[0]
            height = img_open.size[1]
        self.img_Tk = ImageTk.PhotoImage(img_open)
        self.can_src1.create_image(width / 1.92, height / 1.92, image=self.img_Tk, anchor='center')

    # 灰度处理
    def gray_cope(self):
        self.gray_pic_path = self.img_src_path
        print(self.gray_pic_path)
        print(type(self.gray_pic_path))
        img_src = cv2.imdecode(np.fromfile(self.gray_pic_path, dtype=np.uint8), -1)  # 从中文路径读取时用
        print(img_src)
        print(type(img_src))
        img_src = Image.fromarray(cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY))

        if img_src.size[0] * img_src.size[1] > 240 * 80:
            img_src = img_src.resize((190, 130), Image.ANTIALIAS)
            w, h = img_src.size[0], img_src.size[1]
        self.img_Tk2 = ImageTk.PhotoImage(img_src)
        self.can_src2.create_image(w / 1.92, h / 1.92, image=self.img_Tk2, anchor='center')

    # 边缘检测
    def edge_detect(self):
        self.edge_detect_pic = self.gray_pic_path
        img_src = cv2.imdecode(np.fromfile(self.edge_detect_pic, dtype=np.uint8), -1)  # 从中文路径读取时用
        img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
        Sobel_x = cv2.Sobel(img_src, cv2.CV_16S, 1, 0)
        absX_img = cv2.convertScaleAbs(Sobel_x)  # 转回uint8
        ret, image = cv2.threshold(absX_img, 0, 255, cv2.THRESH_OTSU)
        img_src = Image.fromarray(image)
        if img_src.size[0] * img_src.size[1] > 240 * 80:
            img_src = img_src.resize((190, 130), Image.ANTIALIAS)
            w, h = img_src.size[0], img_src.size[1]
        self.img_Tk4 = ImageTk.PhotoImage(img_src)
        self.can_src4.create_image(w / 1.92, h / 1.92, image=self.img_Tk4, anchor='center')

    # 使用训练好的unet二值化处理
    def two_value(self):
        self.pre_site_pic = self.img_src_path
        img_src, img_mask = unet_predict(self.unet, self.pre_site_pic)
        # cv2.imshow('img2',img_mask)
        img_src = Image.fromarray(img_mask)
        if img_src.size[0] * img_src.size[1] >= 220 * 60:
            img_src = img_src.resize((220, 60), Image.ANTIALIAS)
            w, h = img_src.size[0], img_src.size[1]
        self.img_Tk5 = ImageTk.PhotoImage(img_src)
        self.can_src5.create_image(w / 1.94, h / 1.8, image=self.img_Tk5, anchor='center')
        # return img_src, img_mask

    # 车牌定位于提取
    def lic_position(self):
        lic_positon_pic = self.img_src_path
        # img_src, img_mask = self.two_value()
        img_src, img_mask = unet_predict(self.unet, lic_positon_pic)
        img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)
        # img_src = Image.fromarray(img_src_copy[:, :, ::-1])  # img_src_copy[:, :, ::-1]将BGR转为RGB
        img_src = Image.fromarray(Lic_img[0][:, :, ::-1])
        if img_src.size[0] * img_src.size[1] >= 220 * 60:
            img_src = img_src.resize((220, 60), Image.ANTIALIAS)
            w, h = img_src.size[0], img_src.size[1]
        self.img_Tk6 = ImageTk.PhotoImage(img_src)
        self.can_src6.create_image(w / 1.94, h / 1.8, image=self.img_Tk6, anchor='center')

    def video_stream(self):
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_frame.imgtk = imgtk
        self.image_frame.configure(image=imgtk)
        self.image_frame.after(1, self.video_stream)
        win.mainloop()
    # video_stream()
    # # 退出摄像头
    # def quit_(self, win, process):
    #     process.terminate()
    #     win.destroy()
    #
    # def quit_(self, root, process):
    #     process.terminate()
    #     root.destroy()
    #
    # def update_image(self, image_label, queue):
    #     frame = queue.get()
    #     im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     a = Image.fromarray(im)
    #     b = ImageTk.PhotoImage(image=a)
    #     image_label.configure(image=b)
    #     image_label._image_cache = b  # avoid garbage collection
    #     self.win.update()
    #
    # def update_all(self, root, image_label, queue):
    #     self.update_image(image_label, queue)
    #     root.after(0, func=lambda: self.update_all(root, image_label, queue))
    #
    # # multiprocessing image processing functions-------------------------------------
    # def image_capture(self, queue):
    #     vidFile = cv2.VideoCapture(0)
    #     while True:
    #         try:
    #             flag, frame = vidFile.read()
    #             if flag == 0:
    #                 break
    #             queue.put(frame)
    #             cv2.waitKey(20)
    #         except:
    #             continue
    #
    # def start_process(self, queue, root, image_label):
    #     global p
    #     p = Process(target=self.image_capture, args=(queue,))
    #     p.start()
    #     root.after(0, func=lambda: self.update_all(root, image_label, queue))

    # 车牌识别
    def lic_display(self):
        if self.img_src_path == None:  # 还没选择图片就进行预测
            self.can_src1.create_text(18, 50, text='请选择图片', anchor='nw', font=('黑体', 24))
        else:
            img_src = cv2.imdecode(np.fromfile(self.img_src_path, dtype=np.uint8), -1)  # 从中文路径读取时用
            h, w = img_src.shape[0], img_src.shape[1]
            if h * w <= 240 * 80 and 2 <= w / h <= 5:  # 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
                lic = cv2.resize(img_src, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :, :3]  # 直接resize为(240,80)
                img_src_copy, Lic_img = img_src, [lic]
            else:  # 否则就需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
                img_src, img_mask = unet_predict(self.unet, self.img_src_path)
                img_src_copy, Lic_img = locate_and_correct(img_src,
                                                           img_mask)  # 利用core.py中的locate_and_correct函数进行车牌定位和矫正
            Lic_pred = cnn_predict(self.cnn, Lic_img)  # 利用cnn进行车牌的识别预测,Lic_pred中存的是元祖(车牌图片,识别结果)
            if Lic_pred:
                img_src = Image.fromarray(img_src_copy[:, :, ::-1])  # img_src_copy[:, :, ::-1]将BGR转为RGB
                if img_src.size[0] * img_src.size[1] >= 220 * 130:
                    img_src = img_src.resize((220, 130), Image.ANTIALIAS)
                    w, h = img_src.size[0], img_src.size[1]
                    self.img_Tk = ImageTk.PhotoImage(img_src)
                    self.can_src1.delete('all')  # 显示前,先清空画板
                    self.can_src1.create_image(w / 1.94, h / 1.94, image=self.img_Tk,
                                               anchor='center')  # img_src_copy上绘制出了定位的车牌轮廓,将其显示在画板上
                for i, lic_pred in enumerate(Lic_pred):
                    if i == 0:
                        self.lic_Tk = ImageTk.PhotoImage(Image.fromarray(lic_pred[0][:, :, ::-1]))
                        # self.can_src8.create_image(5, 5, image=self.lic_Tk, anchor='nw')
                        self.can_src7.create_text(24, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))
            else:  # Lic_pred为空说明未能识别
                self.can_src1.delete('all')  # 显示前,先清空画板
                self.can_src1.create_text(38, 48, text='未能识别', anchor='nw', font=('黑体', 27))

    # 一键清空
    def clear(self):
        self.can_src1.delete('all')
        self.can_src2.delete('all')
        self.can_src4.delete('all')
        self.can_src5.delete('all')
        self.can_src6.delete('all')
        self.can_src7.delete('all')
        self.img_src_path = None

    def closeEvent(self):  # 关闭前清除session(),防止'NoneType' object is not callable
        self.clear()
        keras.backend.clear_session()
        sys.exit()


if __name__ == '__main__':
    global p
    queue = Queue()
    win = tk.Tk()
    width = 1000  # 窗口宽设定1000
    height = 600  # 窗口高设定600
    Window(win, width, height)
    win.resizable(width=False, height=False)
    win.protocol("WM_DELETE_WINDOW", Window.closeEvent)

    win.mainloop()
