
import numpy as np
import onnxruntime as rt
import cv2
import time
import os
from yolo_helper import draw_image, decode_outputs


def checkModelExtension(fp):
    # Split the extension from the path and normalise it to lowercase.
    ext = os.path.splitext(fp)[-1].lower()

    # Now we can simply use != to check for inequality, no need for wildcards.
    if(ext != ".onnx"):
        raise Exception(fp, "is an unknown file format. Use the model ending with .onnx format")

    if not os.path.exists(fp):
        raise Exception("[ ERROR ] Path of the onnx model file is Invalid")


def checkVideoFileExtension(fp):
    # Split the extension from the path and normalise it to lowercase.
    ext = os.path.splitext(fp)[-1].lower()
    # Now we can simply use != to check for inequality, no need for wildcards.

    if(ext == ".mp4" or ext == ".avi" or ext == ".mov"):
        pass
    else:
        raise Exception(
            fp, "is an unknown file format. Use the video file ending with .mp4 or .avi or .mov formats")

    if not os.path.exists(fp):
        raise Exception("[ ERROR ] Path of the video file is Invalid")


model_file_path = "tiny_yolo_v2_zoo_model.onnx"

# Validate model file path
checkModelExtension('/Users/lisa/Documents/nncase/tmp/ppyolo_tiny_320.onnx')

# Load the model
sess = rt.InferenceSession('/Users/lisa/Documents/nncase/tmp/ppyolo_tiny_320.onnx')

# Get the input name of the model
input_name = sess.get_inputs()[0].name

device = 'CPU_FP32'
# Set OpenVINO as the Execution provider to infer this model
sess.set_providers(['CPUExecutionProvider'], [{'device_type': device}])
'''
other 'device_type' options are: (Any hardware target can be assigned if you have the access to it)
'CPU_FP32', 'GPU_FP32', 'GPU_FP16', 'MYRIAD_FP16', 'VAD-M_FP16', 'VAD-F_FP32',
'HETERO:MYRIAD,CPU',  'MULTI:MYRIAD,GPU,CPU'
'''
frame = cv2.imread('/Users/lisa/Documents/nncase/examples/20classes_yolo/images/dog.bmp')
img_hw = np.array([320, 320], np.uint32)


in_frame = cv2.resize(frame, (320, 320))
X = in_frame[..., [2, 1, 0]]
X = (((X / 255.0) - np.array([0.485, 0.456, 0.406])) /
     np.array([0.229, 0.224, 0.225]))
X = X.transpose(2, 0, 1).astype('float32')
# Reshaping the input array to align with the input shape of the model
X = X[None, ...]
# Running the session by passing in the input data of the model
print(X.dtype)
outputs = sess.run(None, {'image': X})
for output in outputs:
    print(output.shape)

downsample_ratios = [32, 16, 8]
output_shapes = [[1, 3, 85] + (img_hw / ratio).astype(np.int).tolist()
                 for ratio in downsample_ratios]
outputs = [output.reshape(*shape) for output, shape in zip(outputs, output_shapes)]
outputs = [np.transpose(output, [0, 3, 4, 1, 2]) for output in outputs]
anchors = [
    [[220, 125], [128, 222], [264, 266]],
    [[35, 87], [102, 96], [60, 170]],
    [[10, 15], [24, 36], [72, 42]],
]
num_classes = 80
scale_x_y = 1.05
obj_thresh = 0.5
nms_threshold = 0.3

box, clss, score = decode_outputs(outputs[0:1],
                                  anchors[0:1],
                                  downsample_ratios[0:1],
                                  num_classes,
                                  img_hw,
                                  obj_thresh,
                                  nms_threshold,
                                  scale_x_y)


draw_image(frame, box, clss, score)
finall_result = np.concatenate([box, clss[:, None], score[:, None]], -1)
print(finall_result)
