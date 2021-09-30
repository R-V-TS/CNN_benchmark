# import model from module

import cv2
import os
from utils import batch_loop
from CNN_benchmark.utils.load_annotation import LoadDataset
import numpy as np
from CNN_benchmark.draw import drawMatrix
from CNN_benchmark.statistical import ConfusionMatrix, IOU
from CNN_benchmark.statistical.IOU import __calc_iou
from yolov5 import YOLOv5

'''For image distortion'''
from image_processing.noises import AWGN, ASCN, Mult, Speckle
'''End'''

'''For image filtering'''
from image_processing.filters import DCT, Frost, Lee, Median
'''End'''

# Load images from dataset
image_path = 'dataset/coco/train2017' # for coco -> 'dataset/voc/JPEGImages'

image_list = []
image_batch = 256 # Count of image for one loop

dataset, classnames = LoadDataset('dataset/all_dataset.json')

'''This path is unique for each model || Load model'''
model = YOLOv5('model_ckpts/yolov5s.pt')
'''End unique path'''

classes = [v for v in classnames.values()]

for images in batch_loop(dataset, image_batch):  ### Start image processing loop
    images_b = [] # batch of images
    annotations = [] # true labels list
    new_images_list = []
    image_loaded = []
    for image in images:  ## Load annotation batch
        try:
            if os.path.exists(image['image_path']):
                image_loaded = cv2.imread(image['image_path'])[:, :, ::-1]
                annotations.append({
                    'box': image['box'],
                    'classname': image['classname']
                })
        except:
            print(f"Annotation for {image['image_path']} not found")

    results = model.predict(image_loaded, size=640)

    y_t = []
    y_p = []

    t_b = {}
    p_b = {}

    for i, (res, labels) in enumerate(zip(results.pred, annotations)):
        t_b[str(i)] = []
        p_b[str(i)] = []
        for [x1, y1, x2, y2, conf, class_id] in res:
            ious = []
            for box in labels['box']:
                ious.append(__calc_iou(box, [int(x1), int(y1), int(x2), int(y2)]))

            iou_id = np.argmax(ious)
            if ious[iou_id] > 0.1:
                if labels['classname'][iou_id] in classes:
                    y_t.append(classes.index(labels['classname'][iou_id]))
                    y_p.append(int(class_id))
                    t_b[str(i)].append(labels['box'][iou_id])
                    p_b[str(i)].append([int(x1), int(y1), int(x2), int(y2)])

conf = ConfusionMatrix.ConfusionMatrix(y_t, y_p, class_length=len(classes))

drawMatrix.ConfusionMatrix(conf, save=True)

print(IOU.IoU(t_b, p_b))

## End image processing loop

