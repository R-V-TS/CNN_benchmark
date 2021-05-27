# import model from module

import cv2
import os
from utils import batch_loop
from CNN_benchmark.utils.coco_load_annotation import LoadDataset
import numpy as np
from CNN_benchmark.draw import drawMatrix
from CNN_benchmark.statistical import ConfusionMatrix, IOU
from CNN_benchmark.statistical.IOU import __calc_iou

'''For image distortion'''
from image_processing.noises import AWGN, ASCN, Mult, Speckle
'''End'''

'''For image filtering'''
from image_processing.filters import DCT, Frost, Lee, Median
'''End'''

# Load images from dataset
image_path = 'dataset/coco/train2017' # for coco -> 'dataset/voc/JPEGImages'

image_list = []
image_batch = 64 # Count of image for one loop

dataset, classnames = LoadDataset('dataset/coco/annotations/instances_train2017.json')

'''This path is unique for each model || Load model'''
model_path = ""
model = YOLOv5(model_path)
'''End unique path'''

classes = [v for v in classnames.values()]



images = [AWGN(image, mu=0, sigma=10)] #numpy array

for images in batch_loop(os.path.join(image_path), image_batch):  ### Start image processing loop
    images_b = [] # batch of images
    annotations = [] # true labels list
    new_images_list = []
    for img_key in images:  ## Load annotation batch
        try:
            annotations.append(dataset[img_key])
            new_images_list.append(img_key)
        except:
            print(f"Annotation for {img_key} not found")

    for i, img_path in enumerate(new_images_list):  ## Load images batch
        try:
            image_loaded = cv2.imread(os.path.join(image_path, img_path))[:, :, ::-1]
            ## if noised: image_loaded = AWGN(image_loaded)
            ## if filtered: image_loaded = DCT(image_loaded)
            images_b.appemd(image_loaded)
        except:
            del annotations[i]

    results = model.predict

    y_t = []
    y_p = []

    t_b = {}
    p_b = {}

    for i, (res, labels) in enumerate(zip(results.pred, annotations)):
        i  # number of image
        res  # array of bbox [x1, y1, x2, y2, conf, class]
        labels[class_id]
        t_b[str(i)] = []
        p_b[str(i)] = []
        for [x1, y1, x2, y2, conf, class_id] in res:
            ious = []
            for lab in labels:
                ious.append(__calc_iou([lab, [x1, y1, x2, y2]]))

            iou_id = np.argmax(ious)
            if ious[iou_id] > 0.1:
                y_t.append(classes.index(labels[iou_id].values()[0]))
                y_p.append(class_id)
                t_b[str(i)].append([labels[iou_id].keys()[0]])
                p_b[str(i)].append([x1, y1, x2, y2])

    conf = ConfusionMatrix.ConfusionMatrix(y_t, y_p)

    drawMatrix.ConfusionMatrix(conf, save=True)

    print(IOU.IoU(t_b, p_b))

## End image processing loop

