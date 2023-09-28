## Supported ONNX ops

## 支持的 ONNX 算子

| Operator                  | Is Supported | Version(onnx)          | Supported Feature | Not Supported Feature | supported optimized or not | Remark   |
|---------------------------|--------------|------------------------|-------------------|-----------------------|----------------------------|----------|
| Abs                       | ✅            | 13, 6, 1               | 支持导入              |                       | ✅                          |          |
| Acos                      | ✅            | 7                      | 支持导入              |                       | ✅                          |          |
| Acosh                     | ✅            | 9                      | 支持导入              |                       | ✅                          |          |
| Add                       | ✅            | 14, 13, 7, 6, 1        | 支持导入              |                       | ✅                          |          |
| ArgMax                    | ✅            | 13, 12, 11, 1          | 支持导入              |                       |                            |          |
| ArgMin                    | ✅            | 13, 12, 11, 1          | 支持导入              |                       |                            |          |
| Asin                      | ✅            | 7                      | 支持导入              |                       | ✅                          |          |
| Asinh                     | ✅            | 9                      | 支持导入              |                       | ✅                          |          |
| Atan                      | ✅            | 7                      |                   | 不支持导入                 | ✅                          |          |
| Atanh                     | ✘            | /                      | /                 | /                     | /                          | /        |
| AveragePool               | ✅            | 19, 11, 10, 7, 1       | 支持导入              |                       |                            |          |
| AffineGrid                | ✘            | /                      | /                 | /                     | /                          |          |
| BatchNormalization        | ✅            | 15, 14, 9, 7, 6, 1     | 支持导入              |                       |                            |          |
| BitShift                  | ✘            | /                      | /                 | /                     | /                          | /        |
| BitwiseAnd                | ✘            | /                      | /                 | /                     | /                          | /        |
| BitwiseNot                | ✘            | /                      | /                 | /                     | /                          | /        |
| BitwiseOr                 | ✘            | /                      | /                 | /                     | /                          | /        |
| BitwiseXor                | ✘            | /                      | /                 | /                     | /                          | /        |
| Bernoulli                 | ✘            | /                      | /                 | /                     | /                          | /        |
| BlackmanWindow            | ✘            | /                      | /                 | /                     | /                          | /        |
| CastLike                  | ✘            | /                      | /                 | /                     | /                          | /        |
| Cast                      | ✅            | 19, 13, 9, 6, 1        | 支持导入              |                       |                            |          |
| Ceil                      | ✅            | 13, 6, 1               | 支持导入              |                       |                            |          |
| Col2Im                    | ✘            | /                      | /                 | /                     | /                          | /        |
| Compress                  | ✘            | /                      | /                 | /                     | /                          | /        |
| Concat                    | ✅            | 13, 11, 4, 1           | 支持导入              |                       |                            |          |
| ConcatFromSequence        | ✘            | /                      | /                 | /                     | /                          | /        |
| Constant                  | ✅            | 19, 13, 12, 11, 9, 1   | 支持导入              |                       |                            |          |
| ConstantOfShape           | ✅            | 20, 9                  | 支持导入              |                       |                            |          |
| Conv                      | ✅            | 11, 1                  | 支持导入              |                       |                            |          |
| ConvInteger               | ✘            | /                      | /                 | /                     | /                          | /        |
| ConvTranspose             | ✅            | 11, 1                  | 支持导入              |                       |                            |          |
| Cos                       | ✅            | 7                      | 支持导入              |                       | ✅                          |          |
| Cosh                      | ✅            | 9                      | 支持导入              |                       | ✅                          |          |
| CumSum                    | ✅            | 14, 11                 | 支持导入              |                       |                            |          |
| Celu                      | ✅            | 12                     | 支持导入              |                       |                            |          |
| CenterCropPad             | ✘            | /                      | /                 | /                     | /                          | /        |
| Clip                      | ✅            | 13, 12, 11, 6, 1       | 支持导入              |                       |                            |          |
| DFT                       | ✘            | /                      | /                 | /                     | /                          | /        |
| DeformConv                | ✘            | /                      | /                 | /                     | /                          | /        |
| DepthToSpace              | ✅            | 13, 11, 1              | 支持导入              |                       |                            |          |
| DequantizeLinear          | ✅            | 19, 13, 10             | 支持导入              |                       |                            |          |
| Det                       | ✘            | /                      | /                 | /                     | /                          | /        |
| Div                       | ✅            | 14, 13, 7, 6, 1        | 支持导入              |                       |                            |          |
| Dropout                   | ✅            | 13, 12, 10, 7, 6, 1    | 支持导入              |                       |                            |          |
| DynamicQuantizeLinear     | ✘            | /                      | /                 | /                     | /                          | /        |
| Einsum                    | ✘            | /                      | /                 | /                     | /                          | /        |
| Elu                       | ✅            | 6, 1                   | 支持导入              |                       |                            |          |
| Equal                     | ✅            | 19, 13, 11, 7, 1       | 支持导入              |                       |                            |          |
| Erf                       | ✅            | 13, 9                  | 支持导入              |                       |                            |          |
| Exp                       | ✅            | 13, 6, 1               | 支持导入              |                       |                            |          |
| Expand                    | ✅            | 13, 8                  | 支持导入              |                       |                            |          |
| Flatten                   | ✅            | 13, 11, 9, 1           | 支持导入              |                       |                            |          |
| Floor                     | ✅            | 13, 6, 1               | 支持导入              |                       |                            |          |
| GRU                       | ✘            | /                      | /                 | /                     | /                          | /        |
| Gelu                      | ✅            | 20                     |                   | 不支持导入                 |                            |          |
| Gather                    | ✅            | 13, 11, 1              | 支持导入              |                       |                            |          |
| GatherElements            | ✅            | 13, 11                 | 支持导入              |                       |                            |          |
| GatherND                  | ✅            | 13, 12, 11             | 支持导入              |                       |                            |          |
| Gemm                      | ✅            | 13, 11, 9, 7, 6, 1     | 支持导入              |                       |                            |          |
| GlobalAveragePool         | ✅            | 1                      | 支持导入              |                       |                            |          |
| GlobalLpPool              | ✘            | /                      | /                 | /                     | /                          | /        |
| GlobalMaxPool             | ✅            | 1                      | 支持导入              |                       |                            | 目前只支持f32 |
| Greater                   | ✘            | /                      | /                 | /                     | /                          | /        |
| GreaterOrEqual            | ✘            | /                      | /                 | /                     | /                          | /        |
| GridSample                | ✘            | /                      | /                 | /                     | /                          | /        |
| GroupNormalization        | ✘            | /                      | /                 | /                     | /                          | /        |
| Hardmax                   | ✅            | 13, 11, 1              | 支持导入              |                       |                            |          |
| HammingWindow             | ✘            | /                      | /                 | /                     | /                          | /        |
| HannWindow                | ✘            | /                      | /                 | /                     | /                          | /        |
| HardSigmoid               | ✅            | 6, 1                   |                   |                       |                            |          |
| HardSwish                 | ✅            | 14                     | 支持导入              |                       |                            |          |
| Identity                  | ✅            | 19, 16, 14, 13, 1      | 支持导入              |                       |                            |          |
| If                        | ✘            | /                      | /                 | /                     | /                          | /        |
| ImageDecoder              | ✘            | /                      | /                 | /                     | /                          | /        |
| InstanceNormalization     | ✅            | 6, 1                   | 支持导入              |                       |                            |          |
| IsInf                     | ✘            | /                      | /                 | /                     | /                          | /        |
| IsNaN                     | ✘            | /                      | /                 | /                     | /                          | /        |
| LRN                       | ✅            | 13, 1                  | 支持导入              |                       |                            |          |
| LSTM                      | ✅            | 14, 7, 1               | 支持导入              |                       |                            |          |
| Less                      | ✘            | /                      | /                 | /                     | /                          | /        |
| Log                       | ✅            | 13, 6, 1               | 支持导入              |                       |                            |          |
| LpNormalization           | ✘            | /                      | /                 | /                     | /                          | /        |
| LpPool                    | ✘            | /                      | /                 | /                     | /                          | /        |
| LayerNormalization        | ✅            | 17                     |                   | 不支持导入                 | ✅                          |          |
| LeakyRelu                 | ✅            | 16, 6, 1               | 支持导入              |                       |                            |          |
| LogSoftmax                | ✅            | 13, 11, 1              | 支持导入              |                       |                            |          |
| LessOrEqual               | ✘            | /                      | /                 | /                     | /                          | /        |
| MatMul                    | ✅            | 13, 9, 1               | 支持导入              |                       |                            |          |
| MatMulInteger             | ✘            | /                      | /                 | /                     | /                          | /        |
| Max                       | ✅            | 13, 12, 8, 6, 1        | 支持导入              |                       |                            |          |
| MaxPool                   | ✅            | 12, 11, 10, 8, 1       | 支持导入              |                       |                            |          |
| MaxRoiPool                | ✘            | /                      | /                 | /                     | /                          | /        |
| MaxUnpool                 | ✘            | /                      | /                 | /                     | /                          | /        |
| Mean                      | ✅            | 13, 8, 6, 1            |                   | 不支持导入                 |                            |          |
| MelWeightMatrix           | ✘            | /                      | /                 | /                     | /                          | /        |
| MeanVarianceNormalization | ✘            | /                      | /                 | /                     | /                          | /        |
| Min                       | ✅            | 13, 12, 8, 6, 1        | 支持导入              |                       |                            |          |
| Mish                      | ✘            | /                      | /                 | /                     | /                          | /        |
| Mod                       | ✅            | 13, 10                 | 支持导入              |                       |                            |          |
| Mul                       | ✅            | 14, 13, 7, 6, 1        | 支持导入              |                       |                            |          |
| Multinomial               | ✘            | /                      | /                 | /                     | /                          | /        |
| Neg                       | ✅            | 13, 6, 1               | 支持导入              |                       |                            |          |
| NonMaxSuppression         | ✘            | /                      | /                 | /                     | /                          | /        |
| NonZero                   | ✘            | /                      | /                 | /                     | /                          | /        |
| Not                       | ✅            | 1                      | 支持导入              |                       |                            |          |
| NegativeLogLikelihoodLoss | ✘            | /                      | /                 | /                     | /                          | /        |
| OneHot                    | ✅            | 11, 9                  | 支持导入              |                       |                            |          |
| Optional                  | ✘            | /                      | /                 | /                     | /                          | /        |
| OptionalGetElement        | ✘            | /                      | /                 | /                     | /                          | /        |
| OptionalHasElement        | ✘            | /                      | /                 | /                     | /                          | /        |
| Or                        | ✅            | 7, 1                   | 支持导入              |                       |                            |          |
| Pad                       | ✅            | 19, 18, 13, 11, 2, 1	  | 支持导入              |                       |                            |          |
| Pow                       | ✅            | 15, 13, 12, 7, 1       | 支持导入              |                       |                            |          |
| PRelu                     | ✅            | 16, 9, 7, 6, 1         | 支持导入              |                       |                            |          |
| QLinearConv               | ✅            | 10                     | 支持导入              |                       |                            |          |
| QLinearMatMul             | ✅            | 10                     | 支持导入              |                       |                            |          |
| QuantizeLinear            | ✅            | 19, 13, 10             | 支持导入              |                       |                            |          |
| RNN                       | ✘            | /                      | /                 | /                     | /                          | /        |
| RandomNormal              | ✅            | 1                      | 支持导入              |                       |                            |          |
| RandomNormalLike          | ✅            | 1                      | 支持导入              |                       |                            |          |
| RandomUniform             | ✅            | 1                      | 支持导入              |                       |                            |          |
| RandomUniformLike         | ✅            | 1                      | 支持导入              |                       |                            |          |
| Reciprocal                | ✘            | /                      | /                 | /                     | /                          | /        |
| ReduceMax                 | ✅            | 20, 18, 13, 12, 11, 1  | 支持导入              |                       |                            |          |
| ReduceMean                | ✅            | 18, 13, 11, 1	         | 支持导入              |                       |                            |          |
| ReduceMin                 | ✅            | 20, 18, 13, 12, 11, 1	 | 支持导入              |                       |                            |          |
| ReduceProd                | ✅            | 18, 13, 11, 1	         | 支持导入              |                       |                            |          |
| ReduceSum                 | ✅            | 13, 11, 1              | 支持导入              |                       |                            |          |
| RegexFullMatch            | ✘            | /                      | /                 | /                     | /                          |  /       |
| Range                     | ✅            | 11                     | 支持导入              |                       |                            |          |
| ReduceL1                  | ✅            | 18, 13, 11, 1          | 支持导入              |                       |                            |          |
| ReduceL2                  | ✅            | 18, 13, 11, 1          | 支持导入              |                       |                            |          |
| ReduceLogSum              | ✅            | 18, 13, 11, 1          | 支持导入              |                       |                            |          |
| ReduceLogSumExp           | ✅            | 18, 13, 11, 1          | 支持导入              |                       |                            |          |
| ReduceSumSquare           | ✅            | 18, 13, 11, 1          | 支持导入              |                       |                            |          |
| Relu                      | ✅            | 14, 13, 6, 1           | 支持导入              |                       |                            |          |
| Reshape                   | ✅            | 19, 14, 13, 5, 1       | 支持导入              |                       |                            |          |
| Resize                    | ✅            | 19, 18, 13, 11, 10     | 支持导入              |                       |                            |          |
| ReverseSequence           | ✅            | 10                     | 支持导入              |                       |                            |          |
| RoiAlign                  | ✘            | /                      | /                 | /                     | /                          | /        |
| Round                     | ✅            | 11                     | 支持导入              |                       |                            |          |
| STFT                      | ✘            | /                      | /                 | /                     | /                          | /        |
| Scan                      | ✘            | /                      | /                 | /                     | /                          | /        |
| Scatter (deprecated)      | ✘            | /                      | /                 | /                     | /                          | /        |
| ScatterElements           | ✘            | /                      | /                 | /                     | /                          | /        |
| ScatterND                 | ✅            | 18, 16, 13, 11         | 支持导入              |                       |                            |          |
| SequenceAt                | ✘            | /                      | /                 | /                     | /                          | /        |
| SequenceConstruct         | ✘            | /                      | /                 | /                     | /                          | /        |
| SequenceEmpty             | ✘            | /                      | /                 | /                     | /                          | /        |
| SequenceErase             | ✘            | /                      | /                 | /                     | /                          | /        |
| SequenceInsert            | ✘            | /                      | /                 | /                     | /                          | /        |
| SequenceLength            | ✘            | /                      | /                 | /                     | /                          | /        |
| Shape                     | ✅            | 19, 15, 13, 1          | 支持导入              |                       |                            |          |
| Sigmoid                   | ✅            | 13, 6, 1               | 支持导入              |                       |                            |          |
| Sign                      | ✅            | 13, 9                  | 支持导入              |                       |                            |          |
| Sin                       | ✅            | 7                      | 支持导入              |                       |                            |          |
| Sinh                      | ✅            | 9                      | 支持导入              |                       |                            |          |
| Size                      | ✅            | 19, 13, 1              | 支持导入              |                       |                            |          |
| Slice                     | ✅            | 13, 11, 10, 1          | 支持导入              |                       |                            |          |
| SpaceToDepth              | ✅            | 13, 1                  | 支持导入              |                       |                            |          |
| Split                     | ✅            | 18, 13, 11, 2, 1       | 支持导入              |                       |                            |          |
| SplitToSequence           | ✘            | /                      | /                 | /                     | /                          | /        |
| Sqrt                      | ✅            | 13, 6, 1               | 支持导入              |                       |                            |          |
| Squeeze                   | ✅            | 13, 11, 1              | 支持导入              |                       |                            |          |
| StringConcat              | ✘            | /                      | /                 | /                     | /                          | /        |
| StringNormalizer          | ✘            | /                      | /                 | /                     | /                          | /        |
| StringSplit               | ✘            | /                      | /                 | /                     | /                          | /        |
| Sub                       | ✅            | 14, 13, 7, 6, 1        | 支持导入              |                       |                            |          |
| Sum                       | ✅            | 13, 8, 6, 1            | 支持导入              |                       |                            |          |
| Selu                      | ✅            | 6, 1                   |                   | 不支持导入                 |                            |          |
| SequenceMap               | ✘            | /                      | /                 | /                     | /                          | /        |
| Shrink                    | ✘            | /                      | /                 | /                     | /                          | /        |
| Softmax                   | ✅            | 13, 11, 1              | 支持导入              |                       |                            |          |
| SoftmaxCrossEntropyLoss   | ✘            | /                      | /                 | /                     | /                          | /        |
| Softplus                  | ✅            | 1                      | 支持导入              |                       |                            |          |
| Softsign                  | ✅            | 1                      | 支持导入              |                       |                            |          |
| Tan                       | ✘            | /                      | /                 | /                     | /                          | /        |
| Tanh                      | ✅            | 13, 6, 1               | 支持导入              |                       |                            |          |
| TfIdfVectorizer           | ✘            | /                      | /                 | /                     | /                          | /        |
| Tile                      | ✅            | 13, 6, 1               | 支持导入              |                       |                            |          |
| TopK                      | ✅            | 11, 10, 1              | 支持导入              |                       |                            |          |
| Transpose                 | ✅            | 13, 1                  | 支持导入              |                       |                            |          |
| Trilu                     | ✅            | 14                     | 支持导入              |                       |                            |          |
| ThresholdedRelu           | ✘            | /                      | /                 | /                     | /                          | /        |
| Unique                    | ✘            | /                      | /                 | /                     | /                          | /        |
| Unsqueeze                 | ✅            | 13, 11, 1              | 支持导入              |                       |                            |          |
| Upsample (deprecated)     | ✅            | 10, 9, 7               | 支持导入              |                       |                            |          |
| Where                     | ✅            | 16, 9                  | 支持导入              |                       |                            |          |
| Xor                       | ✅            | 7, 1                   |                   | 不支持导入                 |                            |          |
