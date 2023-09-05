* **summarize of runtime kernel feature supported&&not-supported**

| runtime kernel op      | feature supported<br/>(compare to onnx op definition) | feature not-supported<br/>(compare to onnx op definition) |
|------------------------|-------------------------------------------------------|-----------------------------------------------------------|
| batch_normalization    | supported                                             | /                                                         |
| layer_norm             | supported                                             | /                                                         |
| binary add             | supported                                             | /                                                         |
| binary sub             | supported                                             | /                                                         |
| binary mul             | supported                                             | /                                                         |
| binary pow             | supported                                             | /                                                         |
| binary logical_and     | supported                                             | /                                                         |
| binary logical_or      | supported                                             | /                                                         |
| binary logical_xor     | supported                                             | /                                                         |
| bitcast                | /                                                     | not_supported                                             |
| broadcast              | supported                                             | /                                                         | 
| clamp                  | supported                                             | /                                                         | 
| concat                 | supported                                             | /                                                         | 
| condition              | /                                                     | not_supported                                             |
| constant_of_shape      | supported                                             | /                                                         | 
| conv2d                 | supported                                             | /                                                         | 
| conv2d_transpose       | supported                                             | /                                                         | 
| expand                 | supported                                             | /                                                         | 
| dequantize             | supported                                             | /                                                         | 
| flatten                | supported                                             | /                                                         | 
| gather                 | supported                                             | /                                                         | 
| gather_elements        | supported                                             | /                                                         | 
| gather_nd              | supported                                             | /                                                         | 
| scatter_nd             | supported                                             | /                                                         | 
| get_item               | supported                                             | /                                                         | 
| instance_normalization | supported                                             | /                                                         | 
| l2_normalization       | /                                                     | not_supported                                             |
| log_softmax            | supported                                             | /                                                         | 
| lp_normalization       | /                                                     | not_supported                                             |
| lrn                    | supported                                             | /                                                         | 
| lstm                   | supported                                             | /                                                         | 
| mat_mul                | supported                                             | /                                                         | 
| normal                 | supported                                             | /                                                         | 
| normal_like            | supported                                             | /                                                         | 
| one_hot                | supported                                             | /                                                         | 
| pad                    | supported                                             | /                                                         | 
| prelu                  | supported                                             | /                                                         | 
| prod                   | /                                                     | not_supported                                             |
| quantize               | supported                                             | /                                                         | 
| quant_param_of         | /                                                     | not_supported                                             |
| range                  | supported                                             | /                                                         | 
| range_of               | /                                                     | not_supported                                             |
| reduce                 | supported                                             | /                                                         | 
| relu6                  | /                                                     | not_supported                                             |
| require                | supported                                             | /                                                         | 
| bucket_pad             | supported                                             | /                                                         | 
| rank                   | supported                                             | /                                                         | 
| index_of               | supported                                             | /                                                         | 
| fix_shape              | /                                                     | not_supported                                             | 
| reshape                | supported                                             | /                                                         | 
| resize_image           | supported                                             | /                                                         | 
| reverse_sequence       | supported                                             | /                                                         | 
| select                 | /                                                     | not_supported                                             |
| shape_of               | supported                                             | /                                                         | 
| size_of                | supported                                             | /                                                         | 
| slice                  | supported                                             | /                                                         | 
| softmax                | supported                                             | /                                                         | 
| space_to_batch         | supported                                             | /                                                         | 
| split                  | supported                                             | /                                                         | 
| squeeze                | supported                                             | /                                                         | 
| stack                  | supported                                             | /                                                         | 
| tile                   | supported                                             | /                                                         | 
| top_k                  | supported                                             | /                                                         | 
| transpose              | supported                                             | /                                                         | 
| trilu                  | supported                                             | /                                                         | 
| uniform                | supported                                             | /                                                         | 
| uniform_like           | supported                                             | /                                                         | 
| unsqueeze              | supported                                             | /                                                         | 
| where                  | supported                                             | /                                                         | 
| unary abs              | supported                                             | /                                                         | 
| unary acos             | supported                                             | /                                                         | 
| unary acosh            | supported                                             | /                                                         | 
| unary asin             | supported                                             | /                                                         | 
| unary asinh            | supported                                             | /                                                         | 
| unary ceil             | supported                                             | /                                                         | 
| unary cos              | supported                                             | /                                                         | 
| unary cosh             | supported                                             | /                                                         | 
| unary exp              | supported                                             | /                                                         | 
| unary floor            | supported                                             | /                                                         | 
| unary log              | supported                                             | /                                                         | 
| unary logical_not      | supported                                             | /                                                         | 
| unary neg              | supported                                             | /                                                         | 
| unary round            | supported                                             | /                                                         | 
| unary rsqrt            | supported                                             | /                                                         | 
| unary sign             | supported                                             | /                                                         | 
| unary sin              | supported                                             | /                                                         | 
| unary sinh             | supported                                             | /                                                         | 
| unary sqrt             | supported                                             | /                                                         | 
| unary square           | supported                                             | /                                                         | 
| unary tanh             | supported                                             | /                                                         | 
| fake_dequantize        | /                                                     | not_supported                                             |
| fake_quantize          | /                                                     | not_supported                                             |