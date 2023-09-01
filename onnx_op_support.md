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
| broadcast              | supported                                             | not_supported                                             | 
| clamp                  | supported                                             | not_supported                                             | 
| concat                 | supported                                             | not_supported                                             | 
| condition              | /                                                     | not_supported                                             |
| constant_of_shape      | supported                                             | not_supported                                             | 
| conv2d                 | supported                                             | not_supported                                             | 
| conv2d_transpose       | supported                                             | not_supported                                             | 
| expand                 | supported                                             | not_supported                                             | 
| dequantize             | supported                                             | not_supported                                             | 
| flatten                | supported                                             | not_supported                                             | 
| gather                 | supported                                             | not_supported                                             | 
| gather_elements        | supported                                             | not_supported                                             | 
| gather_nd              | supported                                             | not_supported                                             | 
| scatter_nd             | supported                                             | not_supported                                             | 
| get_item               | supported                                             | not_supported                                             | 
| instance_normalization | supported                                             | not_supported                                             | 
| l2_normalization       | /                                                     | not_supported                                             |
| log_softmax            | supported                                             | not_supported                                             | 
| lp_normalization       | /                                                     | not_supported                                             |
| lrn                    | supported                                             | not_supported                                             | 
| lstm                   | supported                                             | not_supported                                             | 
| mat_mul                | supported                                             | not_supported                                             | 
| normal                 | supported                                             | not_supported                                             | 
| normal_like            | supported                                             | not_supported                                             | 
| one_hot                | supported                                             | not_supported                                             | 
| pad                    | supported                                             | not_supported                                             | 
| prelu                  | supported                                             | not_supported                                             | 
| prod                   | /                                                     | not_supported                                             |
| quantize               | supported                                             | not_supported                                             | 
| quant_param_of         | /                                                     | not_supported                                             |
| range                  | supported                                             | not_supported                                             | 
| range_of               | /                                                     | not_supported                                             |
| reduce                 | supported                                             | not_supported                                             | 
| relu6                  | /                                                     | not_supported                                             |
| require                | supported                                             | not_supported                                             | 
| bucket_pad             | supported                                             | not_supported                                             | 
| rank                   | supported                                             | not_supported                                             | 
| index_of               | supported                                             | not_supported                                             | 
| fix_shape              | /                                                     | not_supported                                             | 
| reshape                | supported                                             | not_supported                                             | 
| resize_image           | supported                                             | not_supported                                             | 
| reverse_sequence       | supported                                             | not_supported                                             | 
| select                 | /                                                     | not_supported                                             |
| shape_of               | supported                                             | not_supported                                             | 
| size_of                | supported                                             | not_supported                                             | 
| slice                  | supported                                             | not_supported                                             | 
| softmax                | supported                                             | not_supported                                             | 
| space_to_batch         | supported                                             | not_supported                                             | 
| split                  | supported                                             | not_supported                                             | 
| squeeze                | supported                                             | not_supported                                             | 
| stack                  | supported                                             | not_supported                                             | 
| tile                   | supported                                             | not_supported                                             | 
| top_k                  | supported                                             | not_supported                                             | 
| transpose              | supported                                             | not_supported                                             | 
| trilu                  | supported                                             | not_supported                                             | 
| uniform                | supported                                             | not_supported                                             | 
| uniform_like           | supported                                             | not_supported                                             | 
| unsqueeze              | supported                                             | not_supported                                             | 
| where                  | supported                                             | not_supported                                             | 
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