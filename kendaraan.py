from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import imutils
import pyautogui
import sys
import os

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(
        os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(528, 336)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.uploadButton = QtWidgets.QPushButton(self.centralwidget)
        self.uploadButton.setGeometry(QtCore.QRect(350, 150, 171, 31))
        self.uploadButton.setAutoDefault(False)
        self.uploadButton.setDefault(True)
        self.uploadButton.setFlat(False)
        self.uploadButton.setObjectName("uploadButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 0, 401, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(20, 40, 321, 241))
        self.imageLabel.setFocusPolicy(QtCore.Qt.NoFocus)
        self.imageLabel.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.imageLabel.setStyleSheet("QLineEdit{\n"
"    border-color: #000000;\n"
"    border-width: 1px;\n"
"    border-style: solid;\n"
"}")
        self.imageLabel.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.imageLabel.setText("")
        self.imageLabel.setObjectName("imageLabel")
        self.prosesButton = QtWidgets.QPushButton(self.centralwidget)
        self.prosesButton.setGeometry(QtCore.QRect(350, 200, 171, 31))
        self.prosesButton.setDefault(True)
        self.prosesButton.setObjectName("prosesButton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(350, 70, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 528, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.uploadButton.clicked.connect(self.open_image)
        self.prosesButton.clicked.connect(self.process)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.uploadButton.setText(_translate("MainWindow", "Buka File"))
        self.label.setText(_translate("MainWindow", "CITRA DIGITAL KLASIFIKASI KENDARAAN"))
        self.prosesButton.setText(_translate("MainWindow", "Klasifikasi"))
        self.label_2.setText(_translate("MainWindow", "Nama  : Jihan Khazimah\n"
"NIM   : 201851133"))

    def open_image(self):
        global path
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        pixmap = QPixmap(path)
        pixmap2 = pixmap.scaled(341, 241)
        self.imageLabel.setPixmap(pixmap2)

    def process(self):
        thres = 0.45  # Threshold untuk mendeteksi objek
        nms_thres = 0.2

        classNames = []
        classFile = resource_path('coco.names')
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = resource_path('ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
        weightPath = resource_path('frozen_inference_graph.pb')

        net = cv2.dnn_DetectionModel(weightPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        while True:
            img = cv2.imread(path)
            img = imutils.resize(img, width=640)
            classIds, confs, bbox = net.detect(img, confThreshold=thres)
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))
            indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_thres)

            for i in indices:
                i = i[0]
                box = bbox[i]
                if classNames[classIds[i][0]-1] == "sepeda":
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x, y), (x+w, h+y), color = (0,255,255), thickness = 2)
                    cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,255), 2)
                elif classNames[classIds[i][0]-1] == "mobil":
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x, y), (x+w, h+y), color = (0,255,255), thickness = 2)
                    cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,255), 2)
                elif classNames[classIds[i][0]-1] == "sepeda motor":
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x, y), (x+w, h+y), color = (0, 255, 255), thickness = 2)
                    cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 255), 2)
                elif classNames[classIds[i][0]-1] == "pesawat":
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x, y), (x+w, h+y), color = (0,255,255), thickness = 2)
                    cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,255), 2)
                elif classNames[classIds[i][0]-1] == "bus":
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x, y), (x+w, h+y), color = (0,255,255), thickness = 2)
                    cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,255), 2)
                elif classNames[classIds[i][0]-1] == "kereta":
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x, y), (x+w, h+y), color = (0,255,255), thickness = 2)
                    cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,255), 2)
                elif classNames[classIds[i][0]-1] == "truk":
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x, y), (x+w, h+y), color = (0,255,255), thickness = 2)
                    cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,255), 2)
                elif classNames[classIds[i][0]-1] == "kapal":
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x, y), (x+w, h+y), color = (0,255,255), thickness = 2)

            cv2.imshow("Klasifikasi Kendaraan", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            pyautogui.typewrite('q')



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
