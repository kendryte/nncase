
进入static部分,导出ppyolo tiny模型
首先要修改`static/ppdet/modeling/architectures/yolo.py/_inputs_def`的96行,导出`shape=1`
修改`ppdet/modeling/anchor_heads/yolo_head.py`的464行,`return {'outputs0': outputs[0],'outputs1': outputs[1],'outputs2': outputs[2]}`
```
python tools/export_model.py -c configs/ppyolo/ppyolo_tiny.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_tiny.pdparams TestReader.inputs_def.image_shape=[3,224,224] --output_dir inference_model --exclude_nms

python tools/export_model.py -c configs/ppyolo/ppyolo_tiny.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_tiny.pdparams TestReader.inputs_def.image_shape=[3,320,320] --output_dir inference_model/320 --exclude_nms
```
<!-- 下载权重
```
wget ppyolo_tiny_650e_coco.pdparams
``` -->

<!-- 导出ppyolo tiny模型
```
python tools/export_model.py -c configs/ppyolo/ppyolo_tiny_650e_coco.yml \
                             -o weights=ppyolo_tiny_650e_coco.pdparams \
                             TestReader.inputs_def.image_shape=[1.3,224,224] \
                             --output_dir inference_model \
                             --exlcude_nms
``` -->

导出onnx模型
<!-- cp ppyolo_tiny_650e_coco.pdparams inference_model/ppyolo_tiny_650e_coco/  -->
```
paddle2onnx --model_dir inference_model/ppyolo_tiny \
            --model_filename __model__ \
            --params_filename __params__ \
            --opset_version 11 \
            --save_file ppyolo_tiny.onnx

paddle2onnx --model_dir inference_model/320/ppyolo_tiny \
            --model_filename __model__ \
            --params_filename __params__ \
            --opset_version 11 \
            --save_file ppyolo_tiny_320.onnx
```

<!-- $$
\begin{split} 
hardsigmoid(x)= { 
    \begin{aligned} 
        &0, & & \text{if } x \leq -3 \\
        &1, & & \text{if } x \geq 3 \\
        &alpha * x + beta, & & \text{otherwise} 
    \end{aligned}
    }
\end{split}
$$ -->

## ncc for float
### input shape [224, 224]

```sh
ncc compile tmp/ppyolo_tiny_224.onnx tmp/ppyolo/ppyolo_tiny_224.kmodel -i onnx -t k210 --input-mean 0.48 --input-std 0.225  --dump-ir --dump-dir tmp/ppyolo/ppyolo_tiny_224

ncc infer tmp/yolox_nano/ppyolo_tiny_224.kmodel tmp/ppyolo/ppyolo_tiny_224/infer/float --dataset examples/20classes_yolo/images --dataset-format image --input-mean 0.48 --input-std 0.225
```

### input shape [320, 320]

```
ncc compile tmp/ppyolo_tiny_320.onnx tmp/ppyolo/ppyolo_tiny_320.kmodel -i onnx -t k210 --input-mean 0.48 --input-std 0.225 --dump-ir --dump-dir tmp/ppyolo/ppyolo_tiny_320

ncc infer tmp/ppyolo/ppyolo_tiny_320.kmodel tmp/ppyolo/ppyolo_tiny_320/infer/float --dataset examples/20classes_yolo/images --dataset-format image --input-mean 0.48 --input-std 0.225
```

## ncc for quant

### input shape [224, 224]

```sh
ncc compile tmp/ppyolo_tiny_224.onnx tmp/ppyolo/ppyolo_tiny_224_quant.kmodel -i onnx -t k210 --input-mean 0.48 --input-std 0.225 --dataset examples/20classes_yolo/images --calibrate-method l2 --dump-ir --dump-dir tmp/ppyolo/ppyolo_tiny_224_quant

ncc infer tmp/yolox_nano/ppyolo_tiny_224_quant.kmodel tmp/ppyolo/ppyolo_tiny_224_quant/infer/float --dataset examples/20classes_yolo/images --dataset-format image --input-mean 0.48 --input-std 0.225
```

```sh
ncc compile tmp/ppyolo_tiny_224.onnx tmp/ppyolo/ppyolo_tiny_224_quant.kmodel -i onnx -t cpu --input-mean 0.48 --input-std 0.225 --dataset examples/20classes_yolo/images --dump-ir --dump-dir tmp/ppyolo/ppyolo_tiny_224_quant

ncc infer tmp/ppyolo/ppyolo_tiny_224_quant.kmodel tmp/ppyolo/ppyolo_tiny_224_quant/infer/quant --dataset  --dataset-format image --input-mean 0.48 --input-std 0.225
```