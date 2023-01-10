import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T

_to_tensor = T.Compose([T.ToTensor(), ])


def video_reader(video_path, frame_path, step=5,
                 label_model=None, annotation_path=None, device=None):
    video_capture = cv2.VideoCapture(video_path)
    success, frame_src = video_capture.read()
    c = 0
    annotations = []
    while success:
        if c % step == 0:
            cv2.imwrite(
                os.path.join(frame_path, str('%07d' % (1 + c / step)) + '.jpg'),
                frame_src
            )
            if label_model is not None:
                label_model.to(device)
                label_model.eval()
                with torch.no_grad():
                    img = _to_tensor(frame_src).to(device)
                    pred = label_model.forward((img,))[0]
                    for score, label, box in zip(pred['scores'], pred['labels'], pred['boxes']):
                        score = score.cpu().detach().item()
                        label = label.cpu().detach().item()
                        box = box.cpu().detach().int().tolist()
                        if score >= 0.90:
                            annotations.append(
                                ((1 + c / step), label, box[0], box[1],
                                 box[2] - box[0], box[3] - box[1],
                                 score, label, 0, 0)
                            )

        success, frame_src = video_capture.read()
        c += 1

    if len(annotations):
        np.savetxt(annotation_path, annotations, fmt='%d', delimiter=',')


# if __name__ == '__main__':
#     from core.models import fasterrcnn_resnet50_fpn
#     model = fasterrcnn_resnet50_fpn(pretrained=True)
#     video_reader(r'./datasets/dashcam_1_720P.mp4',
#                  r'./datasets/dashcam/sequences/dashcam_1', 10,
#                  model, r'/datasets/dashcam/annotations/dashcam_1.txt', 'cuda')
