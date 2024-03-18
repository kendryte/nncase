// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Runtime.CompilerServices;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Razor.Templating.Core;

namespace Nncase.CodeGen.CPU;

public static class CSourceBuiltn
{
    public const string KernelHeader = @"#pragma once
#include <nncase/ntt/ntt.h>
using namespace nncase::ntt;

";

    public static string CMakeDef(string name)
    {
        var cmakePath = CMakePath(Path.Combine(Path.GetDirectoryName(typeof(CSourceBuiltn).Assembly.Location)!, "Runtime", "src", "cpu_runtime.cmake"));
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/CMakeLists.txt.cshtml", new { CMakePath = cmakePath }).Result;
        return content;
    }

    public static string MakeKernel(string ctype, string kernelImpl)
    {
        return KernelHeader + ctype + kernelImpl;
    }

    public static string MakeMain(TIR.PrimFunction primFunction, IEnumerable<TIR.Buffer> rdataBuffers)
    {
        string init_tensors = string.Join("\n", primFunction.Parameters.ToArray().Select((b, i) =>
        {
            var buffer = (TIR.Buffer)b;
            var size = TensorUtilities.GetSize(b.CheckedShape.ToValueArray(), TensorUtilities.GetStrides(b.CheckedShape.ToValueArray()), 1);
            return $@"    std::span<{buffer.ElemType.ToC()}, {size}> p{buffer.Name}(({buffer.ElemType.ToC()} *)inputs[{i}], {size});
    tensor_view<{buffer.ElemType.ToC()}, {KernelUtility.DimensionsToC(buffer.Dimensions)}, {KernelUtility.StridesToC(buffer.Strides)}> {buffer.Name}(p{buffer.Name});
";
        }).Concat(rdataBuffers.Select(b =>
        {
            var size = TensorUtilities.GetSize(b.CheckedShape.ToValueArray(), TensorUtilities.GetStrides(b.CheckedShape.ToValueArray()), 1);
            return $@"    std::span<{b.ElemType.ToC()}, {size}> p{b.Name}(({b.ElemType.ToC()}*)(rdata + {((IR.TensorConst)b.MemSpan.Start).Value.ToScalar<ulong>()}), {size});
    tensor_view<{b.ElemType.ToC()}, {KernelUtility.DimensionsToC(b.Dimensions)}, {KernelUtility.StridesToC(b.Strides)}> {b.Name}(p{b.Name});";
        })));
        return @$"#include <nncase/ntt/cpu_runtime.h>
#include ""../device.h""
#include ""kernel.h""

extern ""C"" void kernel_entry(nncase_runtime_cpu_mt_t *cpu_mt, uint8_t **inputs, uint8_t *rdata, uint8_t *l1_data) {{
g_cpu_mt = cpu_mt;
{init_tensors}

    {primFunction.Name}({string.Join(", ", primFunction.Parameters.AsValueEnumerable().Select(b => ((TIR.Buffer)b).Name).ToArray().Concat(rdataBuffers.Select(b => b.Name)).ToArray())}, l1_data);
}}";
    }

    private static string CMakePath(string path) =>
        path.Replace("\\", "/", StringComparison.Ordinal);
}
