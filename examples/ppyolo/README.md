# onnx模型导出



进入`paddle detection(v2.1)`的`static`目录,导出ppyolo tiny模型

1.  修改`static/ppdet/modeling/architectures/yolo.py/_inputs_def`的96行,导出`shape=1`

2.  修改`ppdet/modeling/anchor_heads/yolo_head.py`的464行,`return {'outputs0': outputs[0],'outputs1': outputs[1],'outputs2': outputs[2]}`

3.  执行以下命令

```sh   
python tools/export_model.py -c configs/ppyolo/ppyolo_tiny.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_tiny.pdparams TestReader.inputs_def.image_shape=[3,224,224] --output_dir inference_model/224 --exclude_nms
```

4.  导出`onnx`模型(请先安装`paddle2onnx`)

```sh
paddle2onnx --model_dir inference_model/224/ppyolo_tiny \
            --model_filename __model__ \
            --params_filename __params__ \
            --opset_version 11 \
            --save_file ppyolo_tiny_224.onnx
```

5.  利用`onnxsim`简化模型

```sh
python examples/ppyolo/tools/sim_model.py tmp/ppyolo_tiny_224.onnx tmp/ppyolo_tiny_224_sim.onnx
```

# 转换到浮点模型并验证正确性

1.  编译
```sh
ncc compile tmp/ppyolo_tiny_224_sim.onnx tmp/ppyolo/ppyolo_tiny_224_float.kmodel -i onnx -t k210 --input-mean 0.48 --input-std 0.225 --dump-ir --dump-dir tmp/ppyolo/ir/ppyolo_tiny_224_float
```
2. 推理

```sh
ncc infer tmp/ppyolo/ppyolo_tiny_224_float.kmodel tmp/ppyolo/infer/ppyolo_tiny_224_float --dataset examples/20classes_yolo/images --dataset-format image --input-mean 0.48 --input-std 0.225
```

3. 验证结果

```sh
python examples/ppyolo/tools/test_ncc_output.py tmp/ppyolo/infer/ppyolo_tiny_224_float/${image name}.bin examples/20classes_yolo/images
```

## 导出定点模型并验证正确性

### input shape [224, 224]

1.  编译
```sh
ncc compile tmp/ppyolo_tiny_224_sim.onnx tmp/ppyolo/ppyolo_tiny_224_quant.kmodel -i onnx -t k210 --dataset examples/20classes_yolo/images --input-mean 0.48 --input-std 0.225 --calibrate-method l2 --dump-ir --dump-dir tmp/ppyolo/ir/ppyolo_tiny_224_quant
```
2. 推理

```sh
ncc infer tmp/ppyolo/ppyolo_tiny_224_quant.kmodel tmp/ppyolo/infer/ppyolo_tiny_224_quant --dataset examples/facedetect_landmark/images --dataset-format image --input-mean 0.48 --input-std 0.225
```

3. 验证结果

```sh
python examples/ppyolo/tools/test_ncc_output.py tmp/ppyolo/infer/ppyolo_tiny_224_quant/${image name}.bin examples/facedetect_landmark/images
```