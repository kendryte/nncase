#!/usr/bin/env python3
"""
生成NTT pack操作的测试用例
覆盖以下维度:
1. shape类型: fixed/dynamic
2. vector维度: 1D/2D
3. tensor连续性: contiguous/non-contiguous
4. pack轴: 不同维度
"""

import itertools
from typing import List, Tuple

class PackTestGenerator:
    def __init__(self):
        self.test_cases = []
        
    def generate_test_name(self, shape_type, vector_dim, continuity, pack_axis, ndim):
        """生成测试名称"""
        parts = []
        parts.append(shape_type)
        parts.append(f"{vector_dim}D_vector")
        parts.append(continuity)
        parts.append(f"pack_axis_{pack_axis}")
        parts.append(f"{ndim}D")
        return "_".join(parts)
    
    def generate_shape_init(self, shape_type, dims, var_name="shape"):
        """生成shape初始化代码"""
        if shape_type == "fixed":
            dim_strs = [f"{d}" for d in dims]
            return f"ntt::fixed_shape_v<{', '.join(dim_strs)}>"
        else:  # dynamic
            dim_strs = [str(d) for d in dims]
            return f"ntt::make_shape({', '.join(dim_strs)})"
    
    def generate_tensor_init(self, shape_type, dims, continuity, var_name):
        """生成tensor初始化代码"""
        code = []
        shape_expr = self.generate_shape_init(shape_type, dims)
        
        if continuity == "contiguous":
            code.append(f"alignas(32) auto {var_name} = ntt::make_tensor<float>({shape_expr});")
            code.append(f"NttTest::init_tensor({var_name}, min_input, max_input);")
        else:  # non-contiguous
            # 创建一个更大的tensor，然后创建view
            big_dims = dims.copy()
            # 在倒数第二维上扩大 - 需要处理字符串格式
            if len(big_dims) >= 2:
                big_dims[-2] = f"{big_dims[-2]} * 2"
            big_shape_expr = self.generate_shape_init(shape_type, big_dims)
            
            code.append(f"// 创建非连续tensor")
            code.append(f"alignas(32) auto big_tensor = ntt::make_tensor<float>({big_shape_expr});")
            code.append(f"NttTest::init_tensor(big_tensor, min_input, max_input);")
            code.append(f"")
            code.append(f"auto {var_name} = ntt::make_tensor_view<float>(")
            code.append(f"    big_tensor.elements().data(),")
            code.append(f"    {shape_expr},")
            code.append(f"    big_tensor.strides()");
            code.append(f");")
        
        return code
    
    def generate_pack_axes_str(self, axes):
        """生成pack轴的字符串表示"""
        if len(axes) == 1:
            return f"ntt::fixed_shape_v<{axes[0]}>"
        else:
            return f"ntt::fixed_shape_v<{', '.join(map(str, axes))}>"
    
    def generate_ort_reference(self, input_dims, input_dim_names, pack_axes, vector_dims):
        """生成ORT参考实现"""
        code = []
        ndim = len(input_dims)
        
        # 计算reshape后的形状（用于生成代码的字符串）
        reshape_dims_str = []
        dim_idx = 0
        for i in range(ndim):
            if i in pack_axes:
                axis_idx = pack_axes.index(i)
                # 使用字符串表达式而不是计算结果
                reshape_dims_str.append(f"(int64_t)({input_dim_names[i]} / P)")
                reshape_dims_str.append(f"(int64_t)P")
            else:
                reshape_dims_str.append(f"(int64_t){input_dim_names[i]}")
        
        # 生成reshape代码
        code.append("// ORT参考实现")
        code.append("auto ort_input = NttTest::ntt2ort(ntt_input);")
        code.append(f"int64_t reshape_data[] = {{{', '.join(reshape_dims_str)}}};")
        code.append("int64_t reshape_shape[] = {std::size(reshape_data)};")
        code.append("auto ort_type = NttTest::primitive_type2ort_type<int64_t>();")
        code.append("auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,")
        code.append("                         reshape_shape, std::size(reshape_shape));")
        code.append("auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);")
        
        # 生成transpose的permutation
        if len(pack_axes) > 0:
            # 计算permutation
            perm = []
            packed_dims = []
            j = 0
            for i in range(ndim):
                if i in pack_axes:
                    perm.append(j)
                    packed_dims.append(j + 1)
                    j += 2
                else:
                    perm.append(j)
                    j += 1
            perm.extend(packed_dims)
            
            code.append("")
            code.append(f"int64_t perms[] = {{{', '.join(map(str, perm))}}};")
            code.append("auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));")
        else:
            code.append("auto ort_output = reshaped_tensor;")
        
        return code
    
    def generate_test_case(self, shape_type, vector_dim, continuity, pack_axes, ndim):
        """生成单个测试用例"""
        # 设置维度大小
        P = "NTT_VLEN / (sizeof(float) * 8)"
        
        # 根据ndim生成输入维度
        if ndim == 3:
            dims = [1, 77, 3]  # C, H, W
            dim_names = ['C', 'H', 'W']
        elif ndim == 4:
            dims = [2, 8, 4, 4]  # N, C, H, W  
            dim_names = ['N', 'C', 'H', 'W']
        else:
            dims = [2, 8, 4, 4, 2]  # N, C, H, W, D
            dim_names = ['N', 'C', 'H', 'W', 'D']
        
        # 调整维度以适应pack
        vector_dims = []
        for axis in pack_axes:
            if vector_dim == 1:
                dims[axis] *= 8  # P的值
                vector_dims.append(8)
            else:  # 2D vector
                dims[axis] *= 8
                vector_dims.append(8)
        
        test_name = self.generate_test_name(shape_type, vector_dim, continuity, 
                                           "_".join(map(str, pack_axes)), ndim)
        
        code = []
        code.append(f"TEST(PackTestFloat, {test_name}) {{")
        code.append(f"    constexpr size_t P = {P};")
        
        # 定义维度常量
        for i, (name, size) in enumerate(zip(dim_names, dims)):
            if i in pack_axes:
                axis_idx = pack_axes.index(i)
                coefficient = size // vector_dims[axis_idx]
                code.append(f"    constexpr size_t {name}_coefficient = {coefficient};")
                code.append(f"    constexpr size_t {name} = {name}_coefficient * P;")
            else:
                code.append(f"    constexpr size_t {name} = {size};")
        
        code.append("    float min_input = -10.0f;")
        code.append("    float max_input = 10.0f;")
        code.append("")
        
        # 生成输入tensor
        input_dims_expr = [f"{name}" for name in dim_names]
        tensor_init = self.generate_tensor_init(shape_type, input_dims_expr, continuity, "ntt_input")
        code.extend([f"    {line}" for line in tensor_init])
        code.append("")
        
        # 生成输出tensor
        output_dims = []
        for i, name in enumerate(dim_names):
            if i in pack_axes:
                output_dims.append(f"{name} / P")
            else:
                output_dims.append(name)
        
        if vector_dim == 1:
            vector_type = f"ntt::vector<float, P>"
        else:
            vector_type = f"ntt::vector<float, {', '.join(['P'] * len(pack_axes))}>"
        
        output_shape_expr = self.generate_shape_init(shape_type, output_dims)
        code.append(f"    // 创建输出tensor")
        code.append(f"    alignas(32) auto ntt_output1 = ntt::make_tensor<{vector_type}>({output_shape_expr});")
        code.append("")
        
        # 执行pack操作
        pack_axes_str = self.generate_pack_axes_str(pack_axes)
        code.append(f"    // 执行pack操作")
        code.append(f"    ntt::pack(ntt_input, ntt_output1, {pack_axes_str});")
        code.append("")
        
        # 生成ORT参考实现
        if continuity == "non_contiguous":
            # 需要先复制到连续tensor
            code.append("    // 复制到连续tensor用于ORT参考")
            code.append(f"    alignas(32) auto continuous_input = ntt::make_tensor<float>({self.generate_shape_init(shape_type, input_dims_expr)});")
            
            # 生成嵌套循环复制数据
            code.append("    ")
            for i, name in enumerate(dim_names):
                code.append(f"    {'    ' * i}for (size_t {name.lower()} = 0; {name.lower()} < {name}; {name.lower()}++) {{")
            
            indices = [f"{name.lower()}" for name in dim_names]
            code.append(f"    {'    ' * len(dim_names)}continuous_input({', '.join(indices)}) = ntt_input({', '.join(indices)});")
            
            for i in range(len(dim_names)-1, -1, -1):
                code.append(f"    {'    ' * i}}}")
            code.append("")
            
            # 修改ORT使用continuous_input
            ort_ref = self.generate_ort_reference(dims, dim_names, pack_axes, vector_dims)
            ort_ref[1] = "    auto ort_input = NttTest::ntt2ort(continuous_input);"
        else:
            ort_ref = self.generate_ort_reference(dims, dim_names, pack_axes, vector_dims)
        
        code.extend([f"    {line}" for line in ort_ref])
        code.append("")
        
        # 比较结果
        code.append("    // 比较结果")
        code.append(f"    alignas(32) auto ntt_output2 = ntt::make_tensor<{vector_type}>({output_shape_expr});")
        code.append("    NttTest::ort2ntt(ort_output, ntt_output2);")
        code.append("    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));")
        code.append("}")
        code.append("")
        
        return "\n".join(code)
    
    def generate_all_tests(self):
        """生成所有测试组合"""
        shape_types = ["fixed", "dynamic"]
        vector_dims = [1, 2]
        continuities = ["contiguous", "non_contiguous"]
        
        # 为不同维度定义pack轴选项
        pack_axes_options = {
            3: [[2], [1], [0], [0, 1]],  # 3D: 最后维、倒数第二维、第一维、前两维
            4: [[3], [2], [1], [0], [1, 2]],  # 4D
            5: [[4], [3], [2], [1], [0], [2, 3]]  # 5D
        }
        
        code = []
        
        # 生成文件头
        code.append(self.generate_header())
        
        # 生成测试用例
        for ndim in [3, 4]:  # 先生成3D和4D的测试
            for shape_type, vector_dim, continuity in itertools.product(shape_types, vector_dims, continuities):
                for pack_axes in pack_axes_options[ndim]:
                    # 跳过某些不合理的组合
                    if vector_dim == 2 and len(pack_axes) == 1:
                        continue  # 2D vector需要pack至少2个轴
                    if vector_dim == 1 and len(pack_axes) > 1:
                        continue  # 1D vector只能pack 1个轴
                    
                    test_code = self.generate_test_case(shape_type, vector_dim, continuity, pack_axes, ndim)
                    code.append(test_code)
        
        # 生成main函数
        code.append(self.generate_footer())
        
        return "\n".join(code)
    
    def generate_header(self):
        """生成文件头"""
        return '''/* Copyright 2019-2024 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor.h"
#include "nncase/ntt/tensor_traits.h"
#include "nncase/ntt/vector.h"
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

'''
    
    def generate_footer(self):
        """生成文件尾"""
        return '''int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
'''

if __name__ == "__main__":
    generator = PackTestGenerator()
    test_code = generator.generate_all_tests()
    
    # 写入文件
    with open("test_ntt_pack_generated.cpp", "w") as f:
        f.write(test_code)
    
    print("测试文件已生成: test_ntt_pack_generated.cpp") 