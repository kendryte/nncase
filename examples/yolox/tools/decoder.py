import numpy as np

import torch
import torchvision
import argparse


def decode_outputs(outputs, dtype, hw=np.array([224, 224]), model_strides=[8, 16, 32]):
    model_hw = [hw // s for s in model_strides]
    grids = []
    strides = []
    for (hsize, wsize), stride in zip(model_hw, model_strides):
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(dtype)
    strides = torch.cat(strides, dim=1).type(dtype)

    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    return outputs


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def load_prediction(path, num_classes) -> torch.Tensor:
    arr = np.fromfile(path, dtype=np.float32)
    arr = arr.reshape(1, -1, num_classes + 5)
    prediction = torch.from_numpy(arr)
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser("YOLOX k210 Demo!")
    parser.add_argument('bin', default='tmp/yolox_nano_float/dog.bin', help='bin path')
    parser.add_argument("--tsize", type=str, default=None, help="test image size")
    args = parser.parse_args()

    if not args.tsize:
        hw = np.array([224, 224])
    else:
        hw = np.array([args.tsize, args.tsize])

    num_classes = 80
    conf_thre = 0.1
    nms_thre = 0.1
    strides = [8, 16, 32]
    prediction = load_prediction(args.bin, num_classes)
    prediction = decode_outputs(prediction, prediction.dtype, hw, strides)
    output = postprocess(prediction, num_classes, conf_thre, nms_thre)
    print(output)
