import random

from ui import demo
from draw.draw_box import DrawBox
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import cv2
from PIL import Image
import numpy as np
from threading import Thread
from multiprocessing import Process
import time
import copy

img_formats = ["jpg", "bmp", "jpeg", "png", "tif", "tiff"]


class MyValidator(QValidator):
    def __init__(self, min, max):
        super(MyValidator, self).__init__()
        self.min = min
        self.max = max

    def validate(self, input_str, pos_int):
        # solution1
        try:
            if self.min <= int(input_str) <= self.max:  # 输入结果有效
                return (QValidator.Acceptable, input_str, pos_int)
            elif 0 <= int(input_str) <= self.min:  # 结果待定
                return (QValidator.Intermediate, input_str, pos_int)
            else:  # 验证结果为无效，故新输入的文本不会显示在文本框中
                return (QValidator.Invalid, input_str, pos_int)
        except:
            if len(input_str)==0:
                return (QValidator.Intermediate, input_str, pos_int)#solution2
            return (QValidator.Invalid, input_str, pos_int)

    def fixup(self, p_str):  # solution3
        try:               # solution4
            if int(p_str) < self.min:
                return str(self.min)
            return str(self.max)
        except:            # solution4
            return str(self.min)


class VisualizeArea(QMainWindow):
    def __init__(self):
        super(VisualizeArea, self).__init__()

        self.ui = demo.Ui_mainwindow()
        self.ui.setupUi(self)
        self.imageArea = ImageBox()
        # self.ui.horizontalLayout_5.addWidget(self.imageArea)
        # self.ui.horizontalLayout_5.setStretch(0, 1)
        # self.ui.horizontalLayout_5.setStretch(2, 3)
        self.ui.verticalLayout_14.addWidget(self.imageArea)
        # self.ui.verticalLayout_14.setStretch(0, 0)
        # self.ui.verticalLayout_14.setStretch(1, 0)

        # self.ui.visual_label_format_comboBox1.addItems(['Yolo', 'coco', 'voc', '3', '4', '5', '6', '7', '8'])
        self.ui.visual_label_format_comboBox1.addItems(['Yolo'])
        # self.ui.visual_label_format_comboBox2.addItems(['Yolo', 'coco', 'voc', '3', '4', '5', '6', '7', '8'])
        self.ui.visual_label_format_comboBox2.addItems(['Yolo'])
        self.ui.visual_nms_iou_label.setText("NMS IoU 阈值：" +
                                             str(self.ui.visual_nms_iou_horizontalSlider.value() / 100))
        self.ui.visual_nms_conf_label.setText("NMS 置信度阈值：" +
                                              str(self.ui.visual_nms_conf_horizontalSlider.value() / 100))
        self.ui.visual_iou_label.setText("比对 IoU 阈值：" +
                                         str(self.ui.visual_iou_horizontalSlider.value() / 100))

        statusbar_label = QLabel(self)
        statusbar_label.setOpenExternalLinks(True)
        statusbar_label.setText(
            "<a href='https://github.com/businiaoo/visualizeme'>https://github.com/businiaoo/See</a>")

        self.ui.statusBar.addPermanentWidget(statusbar_label)
        self.setWindowIcon(QIcon("./icons/1.jpg"))

        self.initui()

    def initui(self):
        self.ui.visual_label_format_comboBox1.currentIndexChanged.connect(self.change_drawing_strategy)
        self.ui.visual_label_format_comboBox2.currentIndexChanged.connect(self.change_drawing_strategy)

        self.ui.visual_select_img_button.clicked.connect(self.get_dir_path)
        self.ui.visual_select_gt_label_button.clicked.connect(self.get_dir_path)
        self.ui.visual_select_det_label_button.clicked.connect(self.get_dir_path)

        self.ui.visual_show_det_checkbox.stateChanged.connect(self.change_drawing_strategy)
        self.ui.visual_show_gt_checkbox.stateChanged.connect(self.change_drawing_strategy)
        self.ui.visual_TP_only_checkbox.stateChanged.connect(self.change_drawing_strategy)
        self.ui.visual_FP_only_checkbox.stateChanged.connect(self.change_drawing_strategy)
        self.ui.visual_FN_only_checkbox.stateChanged.connect(self.change_drawing_strategy)
        self.ui.visual_nms_iou_horizontalSlider.valueChanged.connect(self.change_drawing_strategy)
        self.ui.visual_nms_conf_horizontalSlider.valueChanged.connect(self.change_drawing_strategy)
        self.ui.visual_iou_horizontalSlider.valueChanged.connect(self.change_drawing_strategy)

        self.ui.visual_image_index_lineedit.editingFinished.connect(self.input_img_index)
        self.ui.visual_next_image_button.clicked.connect(self.next_image)
        self.ui.visual_previous_image_button.clicked.connect(self.previous_image)
        self.ui.visual_auto_save_checkbox.stateChanged.connect(self.get_dir_path)
        self.ui.visual_save_single_image_button.clicked.connect(self.get_dir_path)
        self.ui.visual_save_all_button.clicked.connect(self.get_dir_path)

        self.check_able()

    def get_dir_path(self):
        button = self.sender()
        if button.text() != "翻页自动保存" and button.text() != "保存单张":
            dir_choose = QFileDialog.getExistingDirectory(self, "选取文件夹")  # 起始路径
        else:
            dir_choose = ""

        if button.text() == "选择图像文件":
            self.ui.visual_select_img_lineedit.setText(dir_choose)
            self.imageArea.image_index = 0
            if dir_choose != "":
                files = os.listdir(dir_choose)
                img_paths = [os.path.join(dir_choose, a) for a in files if a.split(".")[-1] in img_formats]
                self.ui.visual_img_number_label.setText("数量: " + str(len(img_paths)))

                # 整数校验器
                print(len(img_paths))
                intValidator = MyValidator(0, len(img_paths)-1)
                self.ui.visual_image_index_lineedit.setValidator(intValidator)

                self.imageArea.img_paths = img_paths
            else:
                self.ui.visual_img_number_label.setText("数量: 0")

        elif button.text() == "选择真值标签文件":
            self.ui.visual_select_gt_label_lineedit.setText(dir_choose)
            label_formats = ""
            if self.ui.visual_label_format_comboBox1.currentText() == "Yolo":
                label_formats = "txt"
            """
            其他格式的标签
            """
            if dir_choose != "":
                files = os.listdir(dir_choose)
                gt_paths = [os.path.join(dir_choose, a) for a in files if a.split(".")[-1] == label_formats]
                self.ui.visual_gt_label_number_label.setText("数量: " + str(len(gt_paths)))
                self.imageArea.gt_label_dir_path = dir_choose
                self.imageArea.gt_format = label_formats
            else:
                self.ui.visual_gt_label_number_label.setText("数量: 0")
                self.imageArea.gt_label_dir_path = None
                self.imageArea.gt_format = None
        elif button.text() == "选择检测结果文件":
            self.ui.visual_select_det_label_lineedit.setText(dir_choose)
            label_formats = ""
            if self.ui.visual_label_format_comboBox2.currentText() == "Yolo":
                label_formats = "txt"
            """
            其他格式的标签
            """
            if dir_choose != "":
                files = os.listdir(dir_choose)
                det_paths = [os.path.join(dir_choose, a) for a in files if a.split(".")[-1] == label_formats]
                self.ui.visual_det_label_number_label.setText("数量: " + str(len(det_paths)))
                self.imageArea.det_label_dir_path = dir_choose
                self.imageArea.det_format = label_formats
            else:
                self.ui.visual_det_label_number_label.setText("数量: 0")
                self.imageArea.det_label_dir_path = None
                self.imageArea.det_format = None
        elif button.text() == "翻页自动保存":
            if self.ui.visual_auto_save_checkbox.isChecked():
                self.imageArea.auto_save = True
                self.imageArea.auto_save_path = QFileDialog.getExistingDirectory(self, "选取文件夹")  # 起始路径
                if self.imageArea.auto_save_path == "":
                    self.imageArea.auto_save = False
                    self.imageArea.auto_save_path = None
                print("self.imageArea.auto_save_path", self.imageArea.auto_save_path)
            else:
                self.imageArea.auto_save = False
                self.imageArea.auto_save_path = None
        elif button.text() == "保存单张":
            self.imageArea.save_single_path, filetype = QFileDialog.getSaveFileName(self, "保存单张图像", "",
                                                                                    "bmp(*.bmp);;jpg(*.jpg)")  # 起始路径
            """
            保存单张图像的代码
            """
            if self.imageArea.save_single_path != "":
                img = self.imageArea.QPixmap2cv2Img(self.imageArea.img)
                save_path = self.imageArea.save_single_path
                cv2.imwrite(save_path, img)
            self.imageArea.save_single_path = None
        elif button.text() == "全部保存":
            self.imageArea.save_all_path = dir_choose

            """
            循环保存所有图像的代码
            """
            if dir_choose != "":
                # save_thread(self)
                # p = Process(target=self.save_thread, args=(self, ))  # target代表去执行一个任务，如果是加括号的话相当于立马就执行了
                # p.start()
                thread = Thread(target=self.save_thread, args=())
                thread.start()
                thread.join()

                # https://zhuanlan.zhihu.com/p/62988456  pyqt5的多线程

        self.check_able()
        self.show_image()

    def save_thread(self):
        aaa = 1
        img_paths = self.imageArea.img_paths
        gt_label_dir_path = self.imageArea.gt_label_dir_path
        det_label_dir_path = self.imageArea.det_label_dir_path
        show_gt = self.imageArea.show_gt
        show_det = self.imageArea.show_det
        nms_iou = self.imageArea.nms_iou
        nms_conf = self.imageArea.nms_conf
        TP_only = self.imageArea.TP_only
        FP_only = self.imageArea.FP_only
        FN_only = self.imageArea.FN_only
        iou = self.imageArea.iou
        format = self.imageArea.gt_format

        for i in range(len(img_paths)):
            gt_label_file_path = None
            det_label_file_path = None
            print("hahah_______")
            img_path = img_paths[i]

            if gt_label_dir_path is not None:
                if os.path.isfile(self.gt_label_dir_path + os.sep +
                                  img_path.split(os.sep)[-1].replace(img_path.split(".")[-1], self.gt_format)):
                    gt_label_dir_path = self.gt_label_dir_path + os.sep + \
                                        img_path.split(os.sep)[-1].replace(img_path.split(".")[-1], self.gt_format)
            if det_label_dir_path is not None:
                if os.path.isfile(self.det_label_dir_path + os.sep +
                                  img_path.split(os.sep)[-1].replace(img_path.split(".")[-1], self.gt_format)):
                    det_label_dir_path = self.det_label_dir_path + os.sep + \
                                         img_path.split(os.sep)[-1].replace(img_path.split(".")[-1], self.gt_format)
            draw_box = DrawBox(img_path, gt_label_file_path, det_label_file_path)

            img, message = draw_box.start_draw(show_gt, show_det, [nms_iou, nms_conf],
                                               [TP_only, FP_only, FN_only], iou)
            save_path = os.path.join(self.imageArea.save_all_path,
                                     img_path.split(os.sep)[-1])
            cv2.imwrite(save_path, img)
            self.ui.visual_save_all_progressBar.setValue(int((i + 1) / len(img_paths) * 100))
        self.imageArea.save_all_path = None

    def change_drawing_strategy(self):
        self.imageArea.gt_format_name = self.ui.visual_label_format_comboBox1.currentText()
        self.imageArea.det_format_name = self.ui.visual_label_format_comboBox2.currentText()

        self.imageArea.show_gt = self.ui.visual_show_gt_checkbox.isChecked()
        self.imageArea.show_det = self.ui.visual_show_det_checkbox.isChecked()
        self.imageArea.TP_only = self.ui.visual_TP_only_checkbox.isChecked()
        self.imageArea.FP_only = self.ui.visual_FP_only_checkbox.isChecked()
        self.imageArea.FN_only = self.ui.visual_FN_only_checkbox.isChecked()

        self.imageArea.nms_iou = self.ui.visual_nms_iou_horizontalSlider.value() / 100
        self.ui.visual_nms_iou_label.setText("NMS IoU 阈值：" + str(round(self.imageArea.nms_iou, 2)))
        self.imageArea.nms_conf = self.ui.visual_nms_conf_horizontalSlider.value() / 100
        self.ui.visual_nms_conf_label.setText("NMS 置信度阈值：" + str(round(self.imageArea.nms_conf, 2)))
        self.imageArea.iou = self.ui.visual_iou_horizontalSlider.value() / 100
        self.ui.visual_iou_label.setText("比对 IoU 阈值：" + str(round(self.imageArea.iou, 2)))

        self.imageArea.auto_save = self.ui.visual_auto_save_checkbox.isChecked()

        print("self.imageArea.gt_format", self.imageArea.gt_format)
        print("self.imageArea.det_format", self.imageArea.det_format)
        print("self.imageArea.show_gt",  self.imageArea.show_gt)
        print("self.imageArea.show_det", self.imageArea.show_det)
        print("self.imageArea.TP_only", self.imageArea.TP_only)
        print("self.imageArea.FP_only", self.imageArea.FP_only)
        print("self.imageArea.FN_only", self.imageArea.FN_only)
        print("self.imageArea.nms_iou", self.imageArea.nms_iou)
        print("self.imageArea.nms_conf", self.imageArea.nms_conf)
        print("self.imageArea.iou", self.imageArea.iou)
        print("self.imageArea.auto_save_path", self.imageArea.auto_save_path)

        print("-"*50)
        self.check_able()
        self.show_image()

    def next_image(self):

        if self.imageArea.image_index < len(self.imageArea.img_paths)-1:
            self.imageArea.image_index += 1
            self.show_image()
        else:
            self.ui.visual_message_textedit.setText(self.imageArea.message +
                                                    '<br><font color="red" size="5">这是最后一张图像！</font><br>')

    def previous_image(self):
        if self.imageArea.image_index >= 1:
            self.imageArea.image_index -= 1
            self.show_image()
        else:
            self.ui.visual_message_textedit.setText(self.imageArea.message +
                                                    '<br><font color="red" size="5">这是第一张图像！</font><br>')

    def input_img_index(self):
        self.imageArea.image_index = int(self.ui.visual_image_index_lineedit.text())
        self.show_image()

    def check_able(self):

        # 可视化区域的使能性检查
        self.ui.visual_select_gt_label_button.setEnabled(True)
        self.ui.visual_label_format_comboBox1.setEnabled(True)
        self.ui.visual_select_det_label_button.setEnabled(True)
        self.ui.visual_label_format_comboBox2.setEnabled(True)
        self.ui.visual_image_index_lineedit.setEnabled(True)
        self.ui.visual_next_image_button.setEnabled(True)
        self.ui.visual_previous_image_button.setEnabled(True)
        self.ui.visual_auto_save_checkbox.setEnabled(True)
        self.ui.visual_save_single_image_button.setEnabled(True)
        self.ui.visual_save_all_button.setEnabled(True)
        self.ui.visual_show_det_checkbox.setEnabled(True)
        self.ui.visual_nms_iou_horizontalSlider.setEnabled(True)
        self.ui.visual_nms_conf_horizontalSlider.setEnabled(True)
        self.ui.visual_show_gt_checkbox.setEnabled(True)
        self.ui.visual_TP_only_checkbox.setEnabled(True)
        self.ui.visual_FP_only_checkbox.setEnabled(True)
        self.ui.visual_FN_only_checkbox.setEnabled(True)
        self.ui.visual_iou_horizontalSlider.setEnabled(True)
        self.imageArea.setEnabled(True)
        if self.ui.visual_select_gt_label_lineedit.text() == "" or \
                self.ui.visual_select_det_label_lineedit.text() == "":
            self.ui.visual_TP_only_checkbox.setEnabled(False)
            self.ui.visual_FP_only_checkbox.setEnabled(False)
            self.ui.visual_FN_only_checkbox.setEnabled(False)
            self.ui.visual_iou_horizontalSlider.setEnabled(False)
        if self.ui.visual_select_gt_label_lineedit.text() == "":
            self.ui.visual_show_gt_checkbox.setEnabled(False)

        if self.ui.visual_select_det_label_lineedit.text() == "":
            self.ui.visual_show_det_checkbox.setEnabled(False)
            self.ui.visual_nms_iou_horizontalSlider.setEnabled(False)
            self.ui.visual_nms_conf_horizontalSlider.setEnabled(False)
        if self.ui.visual_TP_only_checkbox.isChecked() or \
                self.ui.visual_FP_only_checkbox.isChecked() or \
                self.ui.visual_FN_only_checkbox.isChecked():
            self.ui.visual_show_det_checkbox.setChecked(False)
            self.ui.visual_show_det_checkbox.setEnabled(False)

        else:
            self.ui.visual_iou_horizontalSlider.setEnabled(False)
        if self.ui.visual_select_img_lineedit.text() == "" or \
                int(self.ui.visual_img_number_label.text().split(":")[-1]) == 0:
            self.ui.visual_select_gt_label_button.setEnabled(False)
            self.ui.visual_label_format_comboBox1.setEnabled(False)
            self.ui.visual_select_det_label_button.setEnabled(False)
            self.ui.visual_label_format_comboBox2.setEnabled(False)
            self.ui.visual_image_index_lineedit.setEnabled(False)
            self.ui.visual_next_image_button.setEnabled(False)
            self.ui.visual_previous_image_button.setEnabled(False)
            self.ui.visual_auto_save_checkbox.setEnabled(False)
            self.ui.visual_save_single_image_button.setEnabled(False)
            self.ui.visual_save_all_button.setEnabled(False)
            self.ui.visual_show_det_checkbox.setEnabled(False)
            self.ui.visual_nms_iou_horizontalSlider.setEnabled(False)
            self.ui.visual_nms_conf_horizontalSlider.setEnabled(False)
            self.ui.visual_show_gt_checkbox.setEnabled(False)
            self.ui.visual_TP_only_checkbox.setEnabled(False)
            self.ui.visual_FP_only_checkbox.setEnabled(False)
            self.ui.visual_FN_only_checkbox.setEnabled(False)
            self.ui.visual_iou_horizontalSlider.setEnabled(False)
            self.imageArea.setEnabled(False)
        if self.imageArea.save_all_path != None:
            self.ui.visual_save_all_button.setEnabled(False)

    def show_image(self):
        self.ui.visual_image_index_lineedit.setText(str(self.imageArea.image_index))
        self.imageArea.get_image()
        self.ui.visual_message_textedit.setText(self.imageArea.message)


class ImageBox(QWidget):
    def __init__(self):
        super(ImageBox, self).__init__()
        self.start_pos = None
        self.end_pos = None
        self.left_click = False

        self.point = QPoint(0, 0)
        self.img = None
        self.scaled_img = None

        self.gt_format = None
        self.gt_format_name = None
        self.det_format = None
        self.det_format_name = None

        self.img_paths = None
        self.gt_label_dir_path = None
        self.det_label_dir_path = None
        self.image_index = 0

        self.show_gt = False
        self.show_det = False  # 显示检测结果
        self.TP_only = False
        self.FP_only = False
        self.FN_only = False

        self.nms_iou = 0.5
        self.nms_conf = 0.1
        self.iou = 0.5

        self.message = ""

        self.auto_save = False
        self.auto_save_path = None

        self.save_single_path = None
        self.save_all_path = None

    def get_image(self):
        if self.img_paths is not None:
            # self.point = QPoint(0, 0)
            img_path = self.img_paths[self.image_index]
            gt_label_dir_path = None
            det_label_dir_path = None
            if self.gt_label_dir_path is not None:
                if os.path.isfile(self.gt_label_dir_path + os.sep +
                                  img_path.split(os.sep)[-1].replace(img_path.split(".")[-1], self.gt_format)):
                    gt_label_dir_path = self.gt_label_dir_path + os.sep + \
                                  img_path.split(os.sep)[-1].replace(img_path.split(".")[-1], self.gt_format)
            if self.det_label_dir_path is not None:
                if os.path.isfile(self.det_label_dir_path + os.sep +
                                  img_path.split(os.sep)[-1].replace(img_path.split(".")[-1], self.gt_format)):
                    det_label_dir_path = self.det_label_dir_path + os.sep + \
                                  img_path.split(os.sep)[-1].replace(img_path.split(".")[-1], self.gt_format)
            draw_box = DrawBox(img_path, gt_label_dir_path, det_label_dir_path)

            self.img, message = draw_box.start_draw(self.show_gt, self.show_det, [self.nms_iou, self.nms_conf],
                                                    [self.TP_only, self.FP_only, self.FN_only], self.iou)
            self.message = message

            self.img = self.cv2Img2QPixmap(self.img)
            # self.img = QPixmap(self.img_paths[self.image_index])
            if self.auto_save:
                img = self.QPixmap2cv2Img(self.img)  # 需要转成cv2的格式
                save_path = os.path.join(self.auto_save_path, img_path.split(os.sep)[-1])

                cv2.imwrite(save_path, img)

            # self.scaled_img = self.img.scaled(self.size(), Qt.KeepAspectRatio)
            if self.scaled_img is not None:
                self.scaled_img = self.img.scaled(self.scaled_img.width(), self.scaled_img.height(), Qt.KeepAspectRatio)
            else:
                self.scaled_img = self.img.scaled(self.size(), Qt.KeepAspectRatio)

            self.update()

    def paintEvent(self, e):
        """
        receive paint events
        :param e: QPaintEvent
        :return:
        """

        if self.scaled_img:
            painter = QPainter()
            painter.begin(self)

            painter.drawPixmap(self.point, self.scaled_img)

            painter.end()

    def mouseMoveEvent(self, e):
        """
        mouse move events for the widget
        :param e: QMouseEvent
        :return:
        """
        if self.left_click:
            self.end_pos = e.pos() - self.start_pos
            self.point = self.point + self.end_pos
            self.start_pos = e.pos()
            self.repaint()

    def mousePressEvent(self, e):
        """
        mouse press events for the widget
        :param e: QMouseEvent
        :return:
        """
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self.start_pos = e.pos()

    def mouseReleaseEvent(self, e):
        """
        mouse release events for the widget
        :param e: QMouseEvent
        :return:
        """
        if e.button() == Qt.LeftButton:
            self.left_click = False
        elif e.button() == Qt.RightButton:
            self.point = QPoint(0, 0)
            self.scaled_img = self.img.scaled(self.size(), Qt.KeepAspectRatio)
            self.repaint()

    def wheelEvent(self, e):
        w_stride = int(self.img.width() / 10)
        h_stride = int(self.img.height() / 10)
        if e.angleDelta().y() < 0:
            # 缩小图片
            if not (self.scaled_img.width() <= self.img.width() / 5 or
                    self.scaled_img.height() <= self.img.height() / 5):
                self.scaled_img = self.img.scaled(self.scaled_img.width() - w_stride,
                                                  self.scaled_img.height() - h_stride)
                new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / \
                        (self.scaled_img.width() + w_stride)
                new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / \
                        (self.scaled_img.height() + h_stride)
                self.point = QPoint(int(new_w), int(new_h))
                self.repaint()
        elif e.angleDelta().y() > 0:
            # 放大图片
            if not (self.scaled_img.width() >= self.img.width() * 5 or
                    self.scaled_img.height() >= self.img.height() * 5):
                self.scaled_img = self.img.scaled(self.scaled_img.width() + w_stride,
                                                  self.scaled_img.height() + h_stride)
                new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / \
                        (self.scaled_img.width() - w_stride)
                new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / \
                        (self.scaled_img.height() - h_stride)
                self.point = QPoint(int(new_w), int(new_h))
                self.repaint()

    @staticmethod
    def QPixmap2cv2Img(img):
        qimg = QPixmap.toImage(img)
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
        result = result[..., :3]

        return result

    @staticmethod
    def cv2Img2QPixmap(img):
        height, width, bytesPerComponent = img.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        return pixmap
    # def resizeEvent(self, e):
    #     if self.parent is not None:
    #         self.scaled_img = self.img.scaled(self.size())
    #         self.point = QPoint(0, 0)
    #         self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VisualizeArea()
    window.showMaximized()
    window.show()
    sys.exit(app.exec_())