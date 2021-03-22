import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import dlib
from imutils import face_utils
import sys
from PyQt6 import QtGui, QtCore, QtWidgets
import design

class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        #self.pushButton.clicked.connect()


def grab_images( queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 250)
    cap.set(cv2.CAP_PROP_FPS, 30)
    face_detect = dlib.get_frontal_face_detector()
    values = [[], []]
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop_img = gray[y:y + h, x:x + w]
            heartbeat_values = heartbeat_values[1:] + [np.average(crop_img)]
            values[0].append(heartbeat_values)
            heartbeat_times = heartbeat_times[1:] + [time.time()]
            values[1].append(heartbeat_times)
    cap.release()
    return values


class ImageWidget(QtWidgets):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QtWidgets.QStylePainter()
        qp.begin(self)
        if self.image:
            qp.drawImage, self.image)
        qp.end()


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = App()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    sys.exit(app.exec())


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
