import numpy as np
np.set_printoptions(suppress=True)
from scipy.special import expit as sigmoid
import cv2
from matplotlib.pyplot import imshow, show


def calc_xy_offset(out_hw: np.ndarray) -> np.ndarray:
    """ for dynamic sacle get xy offset tensor for loss calc
        Parameters
        ----------
        out_hw : tf.Tensor
        Returns
        -------
        [tf.Tensor]
            xy offset : shape [out h , out w , 1 , 2] type=tf.float32
                              [h, w, : , 2]
        """
    grid_y = np.tile(np.reshape(np.arange(0, out_hw[0]),
                                [-1, 1, 1, 1]), [1, out_hw[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(0, out_hw[1]),
                                [1, -1, 1, 1]), [out_hw[0], 1, 1, 1])
    xy_offset = np.concatenate([grid_x, grid_y], -1)
    return xy_offset.astype(np.float32)


def xywh_to_all(pred_xy: np.ndarray, pred_wh: np.ndarray,
                img_hw: np.ndarray, downsample_ratio: int,
                anchor: np.ndarray,
                scale: float = 1.) -> [np.ndarray, np.ndarray]:
    """decode yolo output box to full image scale

    Args:
        pred_xy (np.ndarray): [b, h, w, anchor_num, 2]
        pred_wh (np.ndarray): [b, h, w, anchor_num, 2]
        img_hw (np.ndarray): [h, w]
        downsample_ratio (int): current net output scale. eg 32
        anchor (np.ndarray): [anchor_num, 2]
    Return:
      all_pred_xy(np.ndarray): [b, h, w, anchor_num, 2]
      all_pred_wh(np.ndarray): [b, h, w, anchor_num, 2]
    """
    bias: float = -0.5 * (scale - 1.)
    gird_hw = (img_hw / downsample_ratio).astype(np.uint32)
    xy_offest = calc_xy_offset(gird_hw)
    gird_wh = gird_hw[::-1]
    img_wh = img_hw[::-1]
    # NOTE the predict value is x,y <==> w,h
    all_pred_xy = (sigmoid(pred_xy * scale + bias) + xy_offest) * (img_wh / gird_wh)
    all_pred_wh = np.exp(pred_wh) * anchor
    return all_pred_xy, all_pred_wh


def center_to_corner(bbox: np.ndarray) -> np.ndarray:
    """convert box coordinate from center to corner
    Parameters
    ----------
    bbox : np.ndarray
        bbox [c_x,c_y,w,h]
    Returns
    -------
    np.ndarray
        bbox [x1,y1,x2,y2]
    """
    x1 = (bbox[..., 0:1] - bbox[..., 2:3] / 2)
    y1 = (bbox[..., 1:2] - bbox[..., 3:4] / 2)
    x2 = (bbox[..., 0:1] + bbox[..., 2:3] / 2)
    y2 = (bbox[..., 1:2] + bbox[..., 3:4] / 2)
    xyxy = np.concatenate([x1, y1, x2, y2], -1)
    return xyxy


def bbox_iou(a: np.ndarray, b: np.ndarray, offset: int = 0, method='iou') -> np.ndarray:
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.
    Parameters
    ----------
    a : np.ndarray
        (n,4) x1,y1,x2,y2
    b : np.ndarray
        (m,4) x1,y1,x2,y2
    offset : int, optional
        by default 0
    method : str, optional
        by default 'iou', can choice ['iou','giou','diou','ciou']
    Returns
    -------
    np.ndarray
        iou (n,m)
    """
    a = a[..., None, :]
    tl = np.maximum(a[..., :2], b[..., :2])
    br = np.minimum(a[..., 2:4], b[..., 2:4])

    area_i = np.prod(np.maximum(br - tl, 0) + offset, axis=-1)
    area_a = np.prod(a[..., 2:4] - a[..., :2] + offset, axis=-1)
    area_b = np.prod(b[..., 2:4] - b[..., :2] + offset, axis=-1)

    if method == 'iou':
        return area_i / (area_a + area_b - area_i)
    else:
        raise NotImplementedError("Not Support other iou method")


def nms_oneclass(bbox: np.ndarray, score: np.ndarray, thresh: float, method='iou') -> np.ndarray:
    """Pure Python NMS oneclass baseline.
    Parameters
    ----------
    bbox : np.ndarray
        bbox, n*(x1,y1,x2,y2)
    score : np.ndarray
        confidence score (n,)
    thresh : float
        nms thresh
    Returns
    -------
    np.ndarray
        keep index
    """
    order = score.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        iou = bbox_iou(bbox[i], bbox[order[1:]], method=method)
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep


colormap = [
    (255, 82, 0), (0, 255, 245), (0, 61, 255), (0, 255, 112), (0, 255, 133),
    (255, 0, 0), (255, 163, 0), (255, 102, 0), (194, 255, 0), (0, 143, 255),
    (51, 255, 0), (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
    (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255), (255, 0, 245),
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
    (61, 230, 250), (255, 6, 51), (11, 102, 255), (255, 7, 71), (255, 9, 224),
    (9, 7, 230), (220, 220, 220), (255, 9, 92), (112, 9, 255), (8, 255, 214),
    (7, 255, 224), (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
    (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7), (255, 122, 8),
    (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255), (235, 12, 255),
    (160, 150, 20), (0, 163, 255), (140, 140, 140), (250, 10, 15), (20, 255, 0),
    (31, 255, 0), (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
    (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255), (11, 200, 200)]


def draw_image(img: np.ndarray, xyxybox: np.ndarray, labels: np.ndarray, scores: np.ndarray, is_show=True) -> np.ndarray:
    """ draw img and show bbox , set ann = None will not show bbox
    """
    for i, a in enumerate(xyxybox):
        label = int(labels[i])
        r_top = tuple(np.maximum(np.minimum(a[0:2], img.shape[1::-1]), 0).astype(int))
        l_bottom = tuple(np.maximum(np.minimum(a[2:], img.shape[1::-1]), 0).astype(int))
        r_bottom = (r_top[0], l_bottom[1])
        org = (np.maximum(np.minimum(r_bottom[0], img.shape[1] - 12), 0),
               np.maximum(np.minimum(r_bottom[1], img.shape[0] - 12), 0))
        cv2.rectangle(img, r_top, l_bottom, colormap[label])
        cv2.putText(img, f'{label} {scores[i]:.2f}',
                    org, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.5, colormap[label], thickness=1)

    if is_show:
        imshow((img).astype('uint8'))
        show()

    return img.astype('uint8')


def decode_outputs(outputs, anchors, downsample_ratios, num_classes, img_hw, obj_thresh, nms_threshold, scale_x_y=1.05):
    """ box list """
    _xyxy_box = []
    _xyxy_box_scores = []
    """ preprocess label """
    for pred_label, anchor, downsample in zip(outputs, anchors, downsample_ratios):
        """ split the label """
        pred_xy = pred_label[..., 0: 2]
        pred_wh = pred_label[..., 2: 4]
        pred_confidence = pred_label[..., 4: 5]
        pred_cls = pred_label[..., 5:]
        if num_classes > 1:
            box_scores = sigmoid(pred_cls) * sigmoid(pred_confidence)
        else:
            box_scores = sigmoid(pred_confidence)

        """ calc pred box  """
        # xywh_to_all will use sigmoid
        pred_xy_A, pred_wh_A = xywh_to_all(
            pred_xy, pred_wh, img_hw, downsample, anchor, scale=scale_x_y)
        # boxes from xywh to xyxy
        boxes = center_to_corner(np.concatenate((pred_xy_A, pred_wh_A), -1))
        boxes = np.reshape(boxes, (-1, 4))
        box_scores = np.reshape(box_scores, (-1, num_classes))
        """ append box and scores to global list """
        _xyxy_box.append(boxes)
        _xyxy_box_scores.append(box_scores)

    xyxy_box = np.concatenate(_xyxy_box, axis=0)
    xyxy_box_scores = np.concatenate(_xyxy_box_scores, axis=0)

    mask = xyxy_box_scores >= obj_thresh

    """ do nms for every classes"""
    _boxes = []
    _scores = []
    _classes = []
    for c in range(num_classes):
        class_boxes = xyxy_box[mask[:, c]]
        class_box_scores = xyxy_box_scores[:, c][mask[:, c]]
        select = nms_oneclass(class_boxes, class_box_scores,
                              nms_threshold, method='iou')
        class_boxes = class_boxes[select]
        class_box_scores = class_box_scores[select]
        _boxes.append(class_boxes)
        _scores.append(class_box_scores)
        _classes.append(np.ones_like(class_box_scores) * c)

    box: np.ndarray = np.concatenate(_boxes, axis=0)
    clss: np.ndarray = np.concatenate(_classes, axis=0)
    score: np.ndarray = np.concatenate(_scores, axis=0)
    return box, clss, score
