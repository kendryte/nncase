nncase
=========================================
`nncase` is a cross-platform neural network optimization toolkit for fast inference.

## NNCase Converter
A tool to convert models between many formats.
### Usage
`ncc -i <input format> -o <output format> [--dataset <dataset path>] <input path> <output path>`

Input formats can be one of `tflite` and `paddle`.

Output formats can be one of `tf`, `tflite` and `k210code`.

### Examples
- Convert TFLite model to K210 code.

  `ncc -i tflite -o k210code --dataset ./images ./mbnetv1.tflite ./mbnetv1.c`

- Convert PaddlePaddle model to TensorFlow model.

  `ncc -i paddle -o tf ./MobileNetV1_pretrained ./mbnetv1.pb`

- Convert PaddlePaddle model to K210 code.

  `ncc -i paddle -o k210code --dataset ./images ./MobileNetV1_pretrained ./mbnetv1.c`

[License (MIT)](https://raw.githubusercontent.com/kendryte/nncase/master/LICENSE)
-------------------------------------------------------------------------------
	MIT License

	Copyright (c) 2018 nncase

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
