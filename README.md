# See

Visualization tools for object detection research

## 1. Function

### 1.1 Visualization

- [x] Visualization of the GT label box(Now only supports rectangular boxes, the rotating box will follow up later), showing the number of various targets in each image.
  - [x] Yolo format (Only this format is now supported)
  - [ ] coco format
  - [ ] voc format
- [x] Visualization of detection results, the confidence threshold and IOU threshold (NMS, **if necessary**) can be adjusted, and the number of detections of various targets can be displayed
- [x] Comparison of GT and detection results, <font color='red'>**show TP FP FN separately**</font>. (**See the targets of false detection and missed detection more quickly, and analyze their causes**)
- [x] Save all visualization results in one click

### 1.2 One-key conversion of label format(stay tuned)

- [ ] yolo to coco
- [ ] yolo to voc
- [ ] coco to yolo
- [ ] voc to yolo
- [ ] coco to voc
- [ ] voc to coco
- [ ] labelme to yolo
- [ ] labelme to coco
- [ ] labelme to voc

### 1.3 Label statistics(stay tuned)

- [ ] Label category and quantity statistics (histogram)
- [ ] Label the area distribution of the frame, which can be classified into categories (histogram)
  - [ ] General statistical histogram
  - [ ] Cluster statistics histogram
- [ ] Label frame aspect ratio distribution, (only for rectangular frame target)
  - [ ] General statistical histogram
  - [ ] Cluster statistics histogram

## 2. Installation

```
# For conda environment(recommend)
1. conda create -n env_name
2. conda activate env_name
3. conda install pyqt5 opencv numpy
4. git clone https://github.com/businiaoo/See.git
5. cd See
6. python See.py

# For pip installation
1. pip install pyqt5 opencv numpy
2. git clone https://github.com/businiaoo/See.git
3. cd See
4. python See.py
```

