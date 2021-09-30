import re


def get_file_header(name: str):
    return f"""## Supported {name} ops

## 支持的 {name} 算子

| Operator | Is Supported |
|-------|------------------ |
"""


if __name__ == "__main__":
    pattern = re.compile("\(([^)]*)\)")
    for name, alias in (
        ('TensorFlow Lite', 'tflite'),
        ('Caffe', 'caffe'),
        ('ONNX', 'onnx'),
        ('PaddlePaddle', 'paddle'),
    ):
        txt = get_file_header(name)
        with open(f'src/importer/{alias}/opcode.def', 'r') as f:
            opcodes = f.read()
            for case in pattern.findall(opcodes):
                txt += f'| {case} | ✅ |\n'
        with open(f'docs/{alias}_ops.md', 'w') as f:
            f.write(txt)
