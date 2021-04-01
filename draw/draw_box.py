# businiaoo 2021/1/30 businiao006@gmail.com
# https://github.com/businiaoo/visualizeme

import cv2
import random
from PIL import Image
from PyQt5.QtGui import *
import numpy as np

img_formats = ["jpg", "bmp", "jpeg", "png", "tif", "tiff"]


class DrawBox:
    """
       Function:
    """

    def __init__(self, img_path, gt_label_path=None, detect_label_path=None, modes=None):
        """
        :param img_path: 图像文件的全路径
        :param gt_label_path: gt标签的全路径
        :param detect_label_path: 检测结果标签的全路径
        :param modes: 由标签格式决定的 flag整数
        :return:
        """
        if modes is None:
            modes = [0, 0]
        self.img_path = img_path
        self.gt_label_path = gt_label_path
        self.detect_label_path = detect_label_path
        self.modes = modes

        random.seed(2)
        self.color_gt = [[84, 255, 159], [205, 92, 92], [0, 191, 255], [0, 245, 255], [255, 236, 139]]
        for _ in range(200):
            self.color_gt.append([random.randint(0, 255) for _ in range(3)])
        random.seed(15)
        self.color_detect = [[191, 62, 255], [255, 69, 0], [255, 110, 180], [0, 139, 139], [139, 69, 19]]
        for _ in range(200):
            self.color_detect.append([random.randint(0, 255) for _ in range(3)])

    def start_draw(self, show_gt, show_det, det_nms_info, TP_FP_FN_only, iou_thres):
        """

        :param show_gt:  布尔类型
        :param show_det:  布尔类型
        :param det_nms_info: NMS的两个阈值 [iou_thres, conf_thres]
        :param TP_FP_FN_only:  是否以 TP,FP,FN的形式展示标签  [True, True, True]
        :param iou_thres: 在判断TP, FP, FN时，所采用的iou阈值
        :return:
        """
        print(self.img_path)
        image = cv2.imread(self.img_path, 3)
        gt_message = ""
        det_message = ""
        tp_fp_fn_message = ""

        if (self.gt_label_path is not None) and show_gt:
            gt_num = {}
            gt_labels = np.loadtxt(self.gt_label_path, dtype='str')
            if len(gt_labels.shape) == 1:
                gt_labels = np.expand_dims(gt_labels, axis=0)
            for line in gt_labels:
                category = str(line[0])
                if category in gt_num.keys():
                    gt_num[category] += 1
                else:
                    gt_num[category] = 1
                text = category

                # 根据mode将标签转为顺时针4点的形式
                (point1, point2, point3, point4) = self.get_four_point(line[1:])

                cv2.line(image, point1, point2, self.color_gt[int(category)], 2, cv2.LINE_AA)
                cv2.line(image, point2, point3, self.color_gt[int(category)], 2, cv2.LINE_AA)
                cv2.line(image, point3, point4, self.color_gt[int(category)], 2, cv2.LINE_AA)
                cv2.line(image, point4, point1, self.color_gt[int(category)], 2, cv2.LINE_AA)
                # 在这里修改标签字体的大小,字体
                text_coord = point1[0], point1[1]-4
                cv2.putText(image, text, text_coord, cv2.FONT_HERSHEY_DUPLEX,
                            0.6, self.color_gt[int(category)], 2)
            all_gt_catrgory = list(gt_num.keys())
            all_gt_catrgory.sort()
            gt_message = '<font size="5">真值标签 Total:{} </font>'.format(sum(list(gt_num.values())))
            for single_catrgory in all_gt_catrgory:
                color = self.dec2hex(self.color_gt[int(single_catrgory)])
                gt_message += '<font color={} size="5">{}:{} </font>'.format(color,
                                                                             single_catrgory, gt_num[single_catrgory])
            gt_message += '<br>'

        # 画检测结果
        if (self.detect_label_path is not None) and show_det:

            # with open(self.detect_label_path, "r") as f:
            #     lines = f.readlines()
            # while "\n" in lines:
            #     lines.remove("\n")
            det_num = {}
            det_labels = np.loadtxt(self.detect_label_path, dtype='str')
            if len(det_labels.shape) == 1:
                det_labels = np.expand_dims(det_labels, axis=0)
            det_labels = self.det_NMS(det_labels, iou_thres=det_nms_info[0], conf_thres=det_nms_info[1])
            for line in det_labels:
                # line = line.split()
                category = str(line[0])
                if category in det_num.keys():
                    det_num[category] += 1
                else:
                    det_num[category] = 1
                conf = str(round(float(line[-1]), 3))
                text = category + " " + conf

                # 根据mode将标签转为顺时针4点的形式
                (point1, point2, point3, point4) = self.get_four_point(line[1:-1])

                cv2.line(image, point1, point2, self.color_detect[int(category)], 2, cv2.LINE_AA)
                cv2.line(image, point2, point3, self.color_detect[int(category)], 2, cv2.LINE_AA)
                cv2.line(image, point3, point4, self.color_detect[int(category)], 2, cv2.LINE_AA)
                cv2.line(image, point4, point1, self.color_detect[int(category)], 2, cv2.LINE_AA)
                # 在这里修改标签字体的大小,字体
                text_coord = point1[0], point1[1]-4
                cv2.putText(image, text, text_coord, cv2.FONT_HERSHEY_DUPLEX,
                            0.6, self.color_detect[int(category)], 2)
            all_gt_catrgory = list(det_num.keys())
            all_gt_catrgory.sort()
            det_message = '<font size="5">检测结果 Total:{} </font>'.format(sum(list(det_num.values())))
            for single_catrgory in all_gt_catrgory:
                color = self.dec2hex(self.color_detect[int(single_catrgory)])
                det_message += '<font color={} size="5">{}:{} </font>'.format(color,
                                                                             single_catrgory, det_num[single_catrgory])
            det_message += '<br>'

        if (self.gt_label_path is not None) and (self.detect_label_path is not None) and any(TP_FP_FN_only):
            gt_labels = np.loadtxt(self.gt_label_path, dtype='str')
            det_labels = np.loadtxt(self.detect_label_path, dtype='str')
            if len(gt_labels.shape) == 1:
                gt_labels = np.expand_dims(gt_labels, axis=0)
            if len(det_labels.shape) == 1:
                det_labels = np.expand_dims(det_labels, axis=0)
            det_labels = self.det_NMS(det_labels, iou_thres=det_nms_info[0], conf_thres=det_nms_info[1])

            tp_color = [0, 255, 0]
            fp_color = [0, 0, 255]
            fn_color = [0, 165, 255]
            tp_num = 0
            fp_num = 0
            fn_num = 0

            i = 0
            while i < gt_labels.shape[0]:
                gt_label = gt_labels[i, :]
                if len(det_labels) != 0:

                    all_iou = self.iou(gt_label[1:].astype(np.float), det_labels[:, 1:-1].astype(np.float))
                    all_iou[det_labels[:, 0] != gt_label[0]] = 0
                    if np.max(all_iou) >= iou_thres:
                        max_index = np.where(all_iou == np.max(all_iou))[0][0]
                        tp_det_label = det_labels[max_index, :]
                        tp_num += 1
                        if TP_FP_FN_only[0]:  # 画出TP
                            category = str(tp_det_label[0])
                            conf = str(round(float(tp_det_label[-1]), 3))
                            text = category + " " + conf + " " + str(round(np.max(all_iou), 3))

                            # 根据mode将标签转为顺时针4点的形式
                            (point1, point2, point3, point4) = self.get_four_point(tp_det_label[1:-1])

                            cv2.line(image, point1, point2, tp_color, 2, cv2.LINE_AA)
                            cv2.line(image, point2, point3, tp_color, 2, cv2.LINE_AA)
                            cv2.line(image, point3, point4, tp_color, 2, cv2.LINE_AA)
                            cv2.line(image, point4, point1, tp_color, 2, cv2.LINE_AA)
                            # 在这里修改标签字体的大小,字体
                            text_coord = point1[0], point1[1] - 4
                            cv2.putText(image, text, text_coord, cv2.FONT_HERSHEY_DUPLEX,
                                        0.6, self.color_detect[int(category)], 2)

                        det_labels = np.delete(det_labels, max_index, axis=0)  # 每次删除一个TP，最后就只剩下FP了
                    else:
                        fn_num += 1
                        if TP_FP_FN_only[2]:  # 画出FN
                            fn_det_label = gt_labels[i, :]
                            category = str(fn_det_label[0])
                            text = category
                            # 根据mode将标签转为顺时针4点的形式
                            (point1, point2, point3, point4) = self.get_four_point(fn_det_label[1:])
                            cv2.line(image, point1, point2, fn_color, 2, cv2.LINE_AA)
                            cv2.line(image, point2, point3, fn_color, 2, cv2.LINE_AA)
                            cv2.line(image, point3, point4, fn_color, 2, cv2.LINE_AA)
                            cv2.line(image, point4, point1, fn_color, 2, cv2.LINE_AA)
                            # 在这里修改标签字体的大小,字体
                            text_coord = point1[0], point1[1] - 4
                            cv2.putText(image, text, text_coord, cv2.FONT_HERSHEY_DUPLEX,
                                        0.6, self.color_gt[int(category)], 2)
                else:
                    fn_num += 1
                    if TP_FP_FN_only[2]:  # 画出FN
                        fn_det_label = gt_labels[i, :]
                        category = str(fn_det_label[0])
                        text = category
                        # 根据mode将标签转为顺时针4点的形式
                        (point1, point2, point3, point4) = self.get_four_point(fn_det_label[1:])
                        cv2.line(image, point1, point2, fn_color, 2, cv2.LINE_AA)
                        cv2.line(image, point2, point3, fn_color, 2, cv2.LINE_AA)
                        cv2.line(image, point3, point4, fn_color, 2, cv2.LINE_AA)
                        cv2.line(image, point4, point1, fn_color, 2, cv2.LINE_AA)
                        # 在这里修改标签字体的大小,字体
                        text_coord = point1[0], point1[1] - 4
                        cv2.putText(image, text, text_coord, cv2.FONT_HERSHEY_DUPLEX,
                                    0.6, self.color_gt[int(category)], 2)
                i += 1
            fp_num = len(det_labels)
            if TP_FP_FN_only[1]:  # 画出FP   删除了所有TP，最后就只剩下FP了

                for fp_det_label in det_labels:
                    all_iou = self.iou(fp_det_label[1:-1].astype(np.float), gt_labels[:, 1:].astype(np.float))
                    category = str(fp_det_label[0])
                    conf = str(round(float(fp_det_label[-1]), 3))
                    text = category + " " + conf + " " + str(round(np.max(all_iou), 3))

                    # 根据mode将标签转为顺时针4点的形式
                    (point1, point2, point3, point4) = self.get_four_point(fp_det_label[1:-1])

                    cv2.line(image, point1, point2, fp_color, 2, cv2.LINE_AA)
                    cv2.line(image, point2, point3, fp_color, 2, cv2.LINE_AA)
                    cv2.line(image, point3, point4, fp_color, 2, cv2.LINE_AA)
                    cv2.line(image, point4, point1, fp_color, 2, cv2.LINE_AA)
                    # 在这里修改标签字体的大小,字体
                    text_coord = point1[0], point1[1] - 4
                    cv2.putText(image, text, text_coord, cv2.FONT_HERSHEY_DUPLEX,
                                0.6, self.color_detect[int(category)], 2)
            tp_fp_fn_message = '<font size="5">混淆矩阵 </font>'
            color = self.dec2hex(tp_color)
            tp_fp_fn_message += '<font color={} size="5">TP:{} </font>'.format(color, tp_num)
            color = self.dec2hex(fp_color)
            print(color)
            tp_fp_fn_message += '<font color={} size="5">FP:{} </font>'.format(color, fp_num)
            color = self.dec2hex(fn_color)
            tp_fp_fn_message += '<font color={} size="5">FN:{} </font>'.format(color, fn_num)

            tp_fp_fn_message += '<br>'

        img_path_message = '<font>当前图像: {}</font><br>'.format(self.img_path) if \
            self.img_path is not None else '<font>当前图像: 未找到 </font><br>'
        gt_path_message = '<font>当前真值标签: {}</font><br>'.format(self.gt_label_path) if\
            self.gt_label_path is not None else '<font>当前真值标签: 未找到 </font><br>'
        det_path_message = '<font>当前检测结果: {}</font><br>'.format(self.detect_label_path) if\
            self.detect_label_path is not None else '<font>当前检测结果: 未找到 </font><br>'
        # if True:
        #     cv2.imshow("aa", image)
        #     cv2.waitKey()
        #     cv2.destroyWindow("aa")
        message = gt_message + det_message + tp_fp_fn_message + '<br>' +\
                  img_path_message + gt_path_message + det_path_message
        # message = gt_message + det_message + tp_fp_fn_message
        return image, message

    def get_four_point(self, line):
        """
         根据 mode得到四点坐标，以左上角为起点
         :param line: str，某一行标签
         :param img_path: str，图像路径
         :return: 返回四点坐标
        """
        img = Image.open(self.img_path)
        w, h = img.size
        # if self.modes[0] == 0:
        #     assert len(line) == 5, "%s 标签长度不等于5" % self.img_path

        line = line.astype(np.float)
        if (line < 0).any() or (line > 1).any():
            print("warning! 坐标超出范围：{}".format(self.img_path))

        point1 = int((line[0] - line[2] / 2) * w), int((line[1] - line[3] / 2) * h)
        point2 = int((line[0] + line[2] / 2) * w), int((line[1] - line[3] / 2) * h)
        point3 = int((line[0] + line[2] / 2) * w), int((line[1] + line[3] / 2) * h)
        point4 = int((line[0] - line[2] / 2) * w), int((line[1] + line[3] / 2) * h)
        return point1, point2, point3, point4

    def det_NMS(self, det_labels, iou_thres, conf_thres):
        """
        不区分类别的 NMS
        在比较 iou与 conf的时候都不包括等于
        :param det_labels:
        :param iou_thres:
        :param conf_thres:
        :return:
        """
        confs = det_labels[:, -1].astype(np.float)
        det_labels = det_labels[confs > conf_thres, :]  # 利用置信度阈值截断

        confs = det_labels[:, -1].astype(np.float)
        a = np.argsort(confs)[::-1]  # 将置信度从大到小排列
        det_labels = det_labels[a, :]  # 改变det_labels的顺序
        bboxes = det_labels[:, 1:-1].astype(np.float)

        i = 0
        while i < len(bboxes):
            all_iou = self.iou(bboxes[i, :], bboxes[i + 1:, :])
            deletes_index = np.array(np.where(all_iou > iou_thres)) + i + 1
            bboxes = np.delete(bboxes, deletes_index, axis=0)
            det_labels = np.delete(det_labels, deletes_index, axis=0)
            i += 1

        return det_labels

    @staticmethod
    def iou(a, b):
        """
        计算 iou
        :param a:
        :param b:
        :return:
        """
        #  out_label[0] != gt_label[0]

        a_area = a[2] * a[3]
        b_area = b[:, 2] * b[:, 3]
        x1 = np.maximum(a[0] - a[2] / 2, b[:, 0] - b[:, 2] / 2)
        y1 = np.maximum(a[1] - a[3] / 2, b[:, 1] - b[:, 3] / 2)
        x2 = np.minimum(a[0] + a[2] / 2, b[:, 0] + b[:, 2] / 2)
        y2 = np.minimum(a[1] + a[3] / 2, b[:, 1] + b[:, 3] / 2)

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        inter = w * h
        iou = inter / (a_area + b_area - inter)

        return iou

    @staticmethod
    def dec2hex(color):
        color_b = hex(color[0])[2:]
        color_b = "0" + color_b if len(color_b) == 1 else color_b
        color_g = hex(color[1])[2:]
        color_g = "0" + color_g if len(color_g) == 1 else color_g
        color_r = hex(color[2])[2:]
        color_r = "0" + color_r if len(color_r) == 1 else color_r
        color = "#" + color_r + color_g + color_b
        return color


if __name__ == "__main__":
    IMG_DIR = r"C:\Users\BuSiniao\Desktop\1\data\images\yd1_0_1.jpeg"
    LABEL_DIR = r"C:\Users\BuSiniao\Desktop\1\data\labels_yolo\yd1_0_1.txt"
    detect_label = r"C:\Users\BuSiniao\Desktop\1\visualizeme\data_sample\det_label\yd1_0_4.txt"
    # aa = DrawBox(IMG_DIR, LABEL_DIR, detect_label)
    aa = DrawBox(IMG_DIR, LABEL_DIR)
    aa.start_draw(True, False, [0.632, 0.1], [True, True, True], 0.05)
    # aa.start_draw(True, True, [0.632, 0.1], [True, True, True], 0.5)
