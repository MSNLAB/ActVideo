import colorsys
import random

import cv2
import numpy as np
from PIL import Image


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def draw_bbox(img, target, object_category):
    colors = ncolors(len(object_category))
    show_image = np.array(img.permute(1, 2, 0) * 255, dtype=np.uint8)
    show_image = cv2.cvtColor(show_image, cv2.COLOR_RGB2BGR)

    boxes = target['boxes']
    labels = target['labels']

    for idx, label in enumerate(labels):
        object_name = object_category[label]
        object_bndbox = boxes[idx]
        x_min = int(object_bndbox[0])
        y_min = int(object_bndbox[1])
        x_max = int(object_bndbox[2])
        y_max = int(object_bndbox[3])

        show_image = cv2.rectangle(
            img=show_image,
            pt1=(x_min, y_min),
            pt2=(x_max, y_max),
            color=colors[label],
            thickness=5
        )

        show_image = cv2.putText(
            img=show_image,
            text=object_name,
            org=(x_min - 5, y_min - 5),
            fontFace=1, fontScale=3,
            color=colors[label],
            thickness=3,
        )
    # cv2.imwrite('D://test.jpg', show_image)
    # cv2.imshow('OpenCV', show_image)
    # cv2.waitKey(0)
    show_image = cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(show_image)
