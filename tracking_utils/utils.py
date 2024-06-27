import os
import cv2
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tlwh_to_tlbr(bboxes_tlwh):
    """
    将边界框从 tlwh 格式转换为 x1y1x2y2 格式。

    参数:
    bboxes_tlwh (np.ndarray): 形状为 (N, 4) 的数组，每行包含一个边界框 [t, l, w, h]

    返回:
    np.ndarray: 形状为 (N, 4) 的数组，每行包含一个边界框 [x1, y1, x2, y2]
    """
    bboxes_tlwh = np.asarray(bboxes_tlwh, dtype=float)
    bboxes_x1y1x2y2 = np.zeros_like(bboxes_tlwh)
    bboxes_x1y1x2y2[:, 0] = bboxes_tlwh[:, 0]  # x1 = t
    bboxes_x1y1x2y2[:, 1] = bboxes_tlwh[:, 1]  # y1 = l
    bboxes_x1y1x2y2[:, 2] = bboxes_tlwh[:, 0] + bboxes_tlwh[:, 2]  # x2 = t + w
    bboxes_x1y1x2y2[:, 3] = bboxes_tlwh[:, 1] + bboxes_tlwh[:, 3]  # y2 = l + h

    return bboxes_x1y1x2y2


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0.0, ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255


    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w / 140.0))
    cv2.putText(im, "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(tlwhs)), (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = "{}".format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ", {}".format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
    return im
