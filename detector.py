import numpy as np
import cv2
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.config import Config

Config.set('graphics', 'window_state', 'maximized')  # окно запускается в развёрнутом виде чтобы исключить
                                                     # возможные засветы из-за других окон на фоне

# инициализация коэффициентов для вычисления ROI (Region Of Interest)
m_diff = 0
x1 = 0.4
x2 = 0.6
y1 = 0.1
y2 = 0.25

# инициализация классификатора, работающего по принципу признаков Хаара
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# объявление элементов интерфейса
root = FloatLayout()
header = Image(pos_hint={"y": .9}, size_hint=(1, .1), color=(.35, .35, .35, 1))
header_label = Label(text='Detector', font_size=25, pos_hint={"y": .9}, size_hint=(1, .1))
background = Image(pos_hint={"y": 0}, size_hint=(1, .9), color=(.17, .17, .17, 1))
camera_frame = Image(pos_hint={"y": .5}, size_hint=(1, .35))
heartbeat_label = Label(text='', font_size=40, pos_hint={"x": .25, "y": .3}, size_hint=(.5, .15), color=(.35, .35, .35, 1))
button = Button(text='', font_size=20, pos_hint={"x": .25, "y": .05}, size_hint=(.5, .15))


# функция, определяющая регион интереса на каждом кадре (должен быть прямоугольник посередние лба)
def getROI(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) > 0:
        img = cv2.rectangle(img, (faces[0][0] + int(x1 * faces[0][2]), faces[0][1] + int(y1 * faces[0][3])),
                            (faces[0][0] + int(x2 * faces[0][2]), faces[0][1] + int(y2 * faces[0][3])), (255, 0, 0), 2)
                            # рисуем рамку вокруг области интереса
                            # (на всякий случай сейвим в переменную вдруг пригодится)

        return [faces[0][0] + int(x1 * faces[0][2]), faces[0][1] + int(y1 * faces[0][3]),  # возвращаем список координат)
                faces[0][0] + int(x2 * faces[0][2]), faces[0][1] + int(y2 * faces[0][3])]
    else:
        return [0, 0, 0, 0]


def getColorSum(frame, color_id):
    return frame[:, :, color_id].sum()


def getColorAverage(frame, color_id):
    return frame[:, :, color_id].sum() * 1.0 / (frame.shape[0] * frame.shape[1])


def upd(frame, bbox, idf, previous_frame, gsums):
    if (idf == 0):
        droi = getROI(frame)
        if (droi[3] > 0):
            bbox = droi

    if (idf > 0):
        df = cv2.absdiff(frame, previous_frame)
        m_diff = 1.0 * df.sum() / (df.shape[0] * df.shape[1])

        if (m_diff > 15):
            droi = getROI(frame)
            if (droi[3] > 0):
                bbox = droi

    roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    green = getColorAverage(roi, 1)  # 2nd channel for Green color
    if (idf > 50):
        gsums.append(green)
    previous_frame = frame
    idf += 1
    return [frame, bbox, idf, previous_frame, gsums, green]


class MyApp(App):
    is_button_pressed = False


    def build(self, **kwargs):
        #kivy
        root.add_widget(header)
        root.add_widget(header_label)
        root.add_widget(background)
        root.add_widget(camera_frame)
        root.add_widget(heartbeat_label)

        button.bind(on_press=self.press_button)
        root.add_widget(button)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.idf = 0
        self.bbox = [0, 0, 10, 10]
        self.gsums = []
        self.previous_frame = self.cap.read()
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return root


    def update(self, dt):
        ret, frame = self.cap.read()
        bbox, idf, previous_frame, gsums = self.bbox, self.idf, self.previous_frame, self.gsums
        lst = upd(frame, bbox, idf, previous_frame, gsums)
        frame, self.bbox, self.idf, self.previous_frame, self.gsums = lst[0], lst[1], lst[2], lst[3], lst[4]
        green = lst[5]
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        if self.is_button_pressed:
            if type(str(int(np.average(frame)))) == None:
                heartbeat_label.text = 'None'
            else:
                heartbeat_label.text = str(int(np.average(green)))
                button.text = "Stop"
        else:
            heartbeat_label.text = ""
            button.text = "Start"
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        camera_frame.texture = texture1


    def press_button(self, instance):
        self.is_button_pressed = not self.is_button_pressed


if __name__ == '__main__':
    MyApp().run()