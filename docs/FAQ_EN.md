## FAQ

### K210 and KPU
1. What's the K210 hardware environment?

    K210 has 6MB general RAM and 2MB KPU dedicated RAM. Your model's input and output featuremaps are stored at 2MB KPU RAM. The weights and other parameters are stored at 6MB RAM.

2. What kinds of ops can be fully accelerated by KPU?

    All the following constraints must be met.
    - Feature map shape: Input feature maps smaller than or equal to 320x240(WxH) and output features map larger than or equal to 4x4(WxH) and channels are between 1 to 1024 are supported.
    - Same symmetric paddings (TensorFlow use asymmetric paddings when stride=2 and size is even).
    - Normal Conv2D and DepthwiseConv2D of 1x1 or 3x3 filters and stride is 1 or 2.
    - MaxPool(2x2 or 4x4) and AveragePool(2x2 or 4x4).
    - Any elementwise activations (ReLU, ReLU6, LeakyRelu, Sigmoid...), PReLU is not supported by KPU.

3. What kinds of ops can be partially accelerated by KPU?

    - Convolutions of asymmetric paddings or valid paddings, nncase will add necessary Pad   and Crop ops around it.
    - Normal Conv2D and DepthwiseConv2D of 1x1 or 3x3 filters but stride is not 1 or 2.   nncase will divide it to KPUConv2D and a StridedSlice (Pad ops maybe necessary).
    - MatMul, nncase will replace it with a Pad(to 4x4)+ KPUConv2D(1x1 filters) + Crop(to 1x1)  .
    - TransposeConv2D, nncase will replace it with a SpaceToBatch + KPUConv2D + BatchToSpace.

### Compile models
1. Fatal: Not supported tflite opcode: DEQUANTIZE

    Use float tflite models, nncase will take care of quantization.

### Deploy models
1. Should I normalize inputs when running the model?

    If it is a uint8 input (ofen quantized model), you don't need normalize instead provide uint8 input (e.g. RGB888 image). If it is a float input, youd should do preprocess.

2. Why I got "KPU allocator cannot allocate more memory"?

    As said in the chapter "K210 and KPU", Your model's input and output featuremaps are stored at 2MB KPU RAM. Every single layer cannot exceed the limit. You can try to reduce the size of the feature maps.

3. Why I got "Out of memory" when running the model?

    When you compile models, nncase will print the working memory needed to run the model. Often the model is loaded to 6MB RAM, so the total main memory usage is the working memory + size of your model.
