* **summarize of runtime kernel feature supported&&not-supported**

| runtime kernel op      | feature supported<br/>(compare to onnx op definition) | feature not-supported<br/>(compare to onnx op definition) |
|------------------------|-------------------------------------------------------|-----------------------------------------------------------|
| batch_normalization    | all supported                                         | /                                                         |
| layer_norm             | all supported                                         | /                                                         |
| binary add             | all supported                                         | /                                                         |
| binary sub             | all supported                                         | /                                                         |
| binary mul             | all supported                                         | /                                                         |
| binary pow             | all supported                                         | /                                                         |
| binary logical_and     | all supported                                         | /                                                         |
| binary logical_or      | all supported                                         | /                                                         |
| binary logical_xor     | all supported                                         | /                                                         |
| bitcast                | /                                                     | not_supported                                             |
| broadcast              | all supported                                         | /                                                         | 
| clamp                  | all supported                                         | /                                                         | 
| cast                   | all supported                                         | /                                                         | 
| concat                 | all supported                                         | /                                                         | 
| condition              | /                                                     | not_supported                                             |
| constant_of_shape      | all supported                                         | /                                                         | 
| conv2d                 | all supported                                         | /                                                         | 
| conv2d_transpose       | all supported                                         | /                                                         | 
| expand                 | all supported                                         | /                                                         | 
| dequantize             | all supported                                         | /                                                         | 
| flatten                | all supported                                         | /                                                         | 
| gather                 | all supported                                         | /                                                         | 
| gather_elements        | all supported                                         | /                                                         | 
| gather_nd              | all supported                                         | /                                                         | 
| scatter_nd             | all supported                                         | /                                                         | 
| get_item               | all supported                                         | /                                                         | 
| instance_normalization | all supported                                         | /                                                         | 
| l2_normalization       | /                                                     | not_supported                                             |
| log_softmax            | all supported                                         | /                                                         | 
| lp_normalization       | /                                                     | not_supported                                             |
| lrn                    | all supported                                         | /                                                         | 
| lstm                   | all supported                                         | /                                                         | 
| mat_mul                | all supported                                         | /                                                         | 
| normal                 | all supported                                         | /                                                         | 
| normal_like            | all supported                                         | /                                                         | 
| one_hot                | all supported                                         | /                                                         | 
| pad                    | all supported                                         | /                                                         | 
| prelu                  | all supported                                         | /                                                         | 
| prod                   | /                                                     | not_supported                                             |
| quantize               | all supported                                         | /                                                         | 
| quant_param_of         | /                                                     | not_supported                                             |
| range                  | all supported                                         | /                                                         | 
| range_of               | /                                                     | not_supported                                             |
| reduce                 | all supported                                         | /                                                         | 
| relu6                  | /                                                     | not_supported                                             |
| require                | all supported                                         | /                                                         | 
| bucket_pad             | all supported                                         | /                                                         | 
| rank                   | all supported                                         | /                                                         | 
| index_of               | all supported                                         | /                                                         | 
| fix_shape              | /                                                     | not_supported                                             | 
| reshape                | all supported                                         | /                                                         | 
| resize_image           | all supported                                         | /                                                         | 
| reverse_sequence       | all supported                                         | /                                                         | 
| select                 | /                                                     | not_supported                                             |
| shape_of               | all supported                                         | /                                                         | 
| size_of                | all supported                                         | /                                                         | 
| slice                  | all supported                                         | /                                                         | 
| softmax                | all supported                                         | /                                                         | 
| space_to_batch         | all supported                                         | /                                                         | 
| split                  | all supported                                         | /                                                         | 
| squeeze                | all supported                                         | /                                                         | 
| stack                  | all supported                                         | /                                                         | 
| tile                   | all supported                                         | /                                                         | 
| top_k                  | all supported                                         | /                                                         | 
| transpose              | all supported                                         | /                                                         | 
| trilu                  | all supported                                         | /                                                         | 
| uniform                | all supported                                         | /                                                         | 
| uniform_like           | all supported                                         | /                                                         | 
| unsqueeze              | all supported                                         | /                                                         | 
| where                  | all supported                                         | /                                                         | 
| unary abs              | reference all supported                               | optimized input's datatype only support float             | 
| unary acos             | reference all supported                               | optimized input's datatype only support float             | 
| unary acosh            | reference all supported                               | optimized input's datatype only support float             | 
| unary asin             | reference all supported                               | optimized input's datatype only support float             | 
| unary asinh            | reference all supported                               | optimized input's datatype only support float             | 
| unary ceil             | reference all supported                               | optimized input's datatype only support float             | 
| unary cos              | reference all supported                               | optimized input's datatype only support float             | 
| unary cosh             | reference all supported                               | optimized input's datatype only support float             | 
| unary exp              | reference all supported                               | optimized input's datatype only support float             | 
| unary floor            | reference all supported                               | optimized input's datatype only support float             | 
| unary log              | reference all supported                               | optimized input's datatype only support float             | 
| unary logical_not      | reference all supported                               | optimized input's datatype only support float             | 
| unary neg              | reference all supported                               | optimized input's datatype only support float             | 
| unary round            | reference all supported                               | optimized input's datatype only support float             | 
| unary rsqrt            | reference all supported                               | optimized input's datatype only support float             | 
| unary sign             | reference all supported                               | optimized input's datatype only support float             | 
| unary sin              | reference all supported                               | optimized input's datatype only support float             | 
| unary sinh             | reference all supported                               | optimized input's datatype only support float             | 
| unary sqrt             | reference all supported                               | optimized input's datatype only support float             | 
| unary square           | reference all supported                               | optimized input's datatype only support float             | 
| unary tanh             | reference all supported                               | optimized input's datatype only support float             | 
| fake_dequantize        | /                                                     | not_supported                                             |
| fake_quantize          | /                                                     | not_supported                                             |