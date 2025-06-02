import numpy as np
import json
import base64
import argparse
import glob
import os
from typing import Dict, Any, List


def numpy_dtype_to_datatype(dtype: np.dtype) -> dict:
    typecodes = {
        np.bool_: 0,
        np.char: 1,
        np.int8: 2,
        np.int16: 3,
        np.int32: 4,
        np.int64: 5,
        np.uint8: 6,
        np.uint16: 7,
        np.uint32: 8,
        np.uint64: 9,
        np.float16: 10,
        np.float32: 11,
        np.float64: 12,
        # np.BFloat16: 13,
        # Float8E4M3: 14,
        # Float8E5M2: 15,
    }

    return {'$type': 'PrimType', 'TypeCode': typecodes[dtype.type]}


def calculate_strides(shape: tuple) -> List[int]:
    """计算C风格的strides"""
    if not shape:
        return []

    strides = [1]
    for i in range(len(shape) - 1, 0, -1):
        strides.insert(0, strides[0] * shape[i])

    return strides


def is_numeric_type(dtype: np.dtype) -> bool:
    """判断是否为数值类型，可以用base64编码"""
    numeric_types = [
        np.float32, np.float64, np.float16,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.bool_
    ]
    return dtype.type in numeric_types


def numpy_to_tensor_json(array: np.ndarray) -> Dict[str, Any]:
    """将numpy数组转换为Tensor JSON格式"""

    # 确保数组是C连续的
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)

    # 获取基本信息
    element_type = numpy_dtype_to_datatype(array.dtype)
    dimensions = list(array.shape)
    strides = calculate_strides(array.shape)

    # 创建JSON结构
    tensor_json = {
        "ElementType": element_type,
        "Dimensions": dimensions,
        "Strides": strides
    }

    buffer_bytes = array.tobytes()
    buffer_base64 = base64.b64encode(buffer_bytes).decode('utf-8')
    tensor_json["Buffer"] = buffer_base64
    return tensor_json


def convert_npy_to_json(npy_file: str, output_dir: str = ".") -> str:
    """转换单个npy文件为json"""
    try:
        # 加载numpy数组
        array = np.load(npy_file)

        # 转换为tensor json格式
        tensor_json = numpy_to_tensor_json(array)

        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(npy_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.json")

        # 写入JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tensor_json, f, indent=2)

        print(f"转换成功: {npy_file} -> {output_file}")
        return output_file

    except Exception as e:
        print(f"转换失败 {npy_file}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="将numpy数组文件转换为Tensor JSON格式")
    parser.add_argument("patterns", nargs="+", help="npy文件模式，例如: input*.npy")
    parser.add_argument("-o", "--output", default=".", help="输出目录 (默认: 当前目录)")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    # 收集所有匹配的文件
    all_files = []
    for pattern in args.patterns:
        files = glob.glob(pattern)
        if not files:
            print(f"警告: 没有找到匹配 '{pattern}' 的文件")
        all_files.extend(files)

    if not all_files:
        print("错误: 没有找到任何npy文件")
        return 1

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)

    # 转换所有文件
    success_count = 0
    total_count = len(all_files)

    for npy_file in all_files:
        if args.verbose:
            print(f"处理: {npy_file}")

        output_file = convert_npy_to_json(npy_file, args.output)
        if output_file:
            success_count += 1

    print(f"\n转换完成: {success_count}/{total_count} 个文件成功")
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    exit(main())
