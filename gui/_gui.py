# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file '_gui.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################
import numpy as np
from PySide2.QtCore import *
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import *
import cv2

from db_connect.query_db import QueryDb
from db_connect.spc import SPC
import matplotlib.pyplot as plt


class Ui_Form(object):
    q: QueryDb

    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1024, 800)
        self.image_label = QLabel(Form)
        self.image_label.setObjectName(u"image_label")
        self.image_label.setGeometry(QRect(250, 100, 800, 600))
        self.comboBox = QComboBox(Form)
        self.q = QueryDb()
        for col in self.q.cols:
            self.comboBox.addItem(col)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(10, 40, 300, 21))
        self.comboBox.activated[str].connect(self.showSPC)
        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)

    # setupUi
    def showSPC(self, param:str):

        data = np.array(self.q.get(param), dtype='float')
        spc = SPC(data)
        x_bar = spc.x_bar
        ucl = spc.ucl
        lcl = spc.lcl
        fig, axs = plt.subplots(1, figsize=(15, 15), sharex=True)
        axs.plot(x_bar, marker='o', color='black')
        axs.plot(ucl, linestyle='dashed', marker='o', color='red')
        axs.plot(lcl, linestyle='dashed', marker='o', color='red')
        axs.plot(data, linestyle='-', marker='o', color='blue')
        spc_img = '../pic/spc.png'
        fig.savefig(spc_img)
        image = cv2.imread(spc_img)
        input_image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)

        height, width, channels = input_image.shape
        bytesPerLine = channels * width
        qImg = QImage(input_image.data, width, height, bytesPerLine, QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(qImg))

        # self.image_label.setText(param)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.image_label.setText(QCoreApplication.translate("Form", u"TextLabel", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("Form", u" ", None))

    # retranslateUi
