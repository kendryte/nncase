import numpy as np
from yolo_helper import draw_image, decode_outputs
import cv2
from pathlib import Path
# 0   save_infer_model/scale_0  f32[1,255,7,7]
# 1   save_infer_model/scale_1  f32[1,255,14,14]
# 2   save_infer_model/scale_2  f32[1,255,28,28]

path = "/Users/lisa/Documents/nncase/tmp/ppyolo/ppyolo_tiny_224_quant/infer/quant/36.bin"
output = np.fromfile(path, dtype=np.float32)


img_hw = np.array([224, 224], np.uint32)
downsample_ratios = [32, 16, 8]
output_shapes = [[1, 3, 85] + (img_hw / ratio).astype(np.int).tolist() for ratio in downsample_ratios]
split_indexs = np.cumsum([np.prod(shape) for shape in output_shapes])
*outputs, _ = np.split(output, split_indexs)
# to anchor nums x 85 --> move axis
outputs = [output.reshape(*shape) for output, shape in zip(outputs, output_shapes)]
outputs = [np.transpose(output, [0, 3, 4, 1, 2]) for output in outputs]
anchors = [
    [[220, 125], [128, 222], [264, 266]],
    [[35, 87], [102, 96], [60, 170]],
    [[10, 15], [24, 36], [72, 42]],
]
num_classes = 80
scale_x_y = 1.05
obj_thresh = 0.1
nms_threshold = 0.1

box, clss, score = decode_outputs(outputs,
                                  anchors,
                                  downsample_ratios,
                                  num_classes,
                                  img_hw,
                                  obj_thresh,
                                  nms_threshold,
                                  scale_x_y)

img_path = (Path("/Users/lisa/Documents/nncase/examples/facedetect_landmark/images") /
            (Path(path).stem + '.jpg'))
img = cv2.imread(str(img_path))
img = cv2.resize(img, tuple(img_hw))
draw_image(img, box, clss, score)
finall_result = np.concatenate([box, clss[:, None], score[:, None]], -1)
print(finall_result)
