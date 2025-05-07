// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Runtime.CompilerServices;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Targets;
using Razor.Templating.Core;

namespace Nncase.CodeGen.NTT;

public record BufferRenderInfo(string Name, string ElemType, int ElemSize, int Rank, ulong Offset, Dimension Size, bool IsFixedDimensions, bool IsFixedStrides, Dimension[] Dimensions, string DimensionsStr, Dimension[] Strides, string StridesStr, string? Distributed)
{
}

public record KernelMainModel(TIR.PrimFunction PrimFunction, TIR.Buffer[] RDataBuffers, NTTTargetOptions Options, ulong Alignment, ulong DataSize, ulong RDataSize, ulong LocalRdataPoolSize)
{
    public BufferRenderInfo GetInfo(TIR.Buffer buffer)
    {
        ulong offset = 0;
        if (buffer.MemSpan.Start is IR.TensorConst tc)
        {
            offset = tc.Value.ToScalar<ulong>();
        }

        var elemType = buffer.ElemType.ToC();
        var rank = buffer.Dimensions.Length;
        var size = buffer.MemSpan.Size / buffer.ElemType.SizeInBytes;
        var isFixedDims = buffer.Dimensions.AsValueEnumerable().All(d => d.IsFixed);
        var isFixedStrides = buffer.Strides.AsValueEnumerable().All(d => d.IsFixed);
        var dims = KernelUtility.DimensionsTypeToC(isFixedDims, buffer.Dimensions);
        var strides = KernelUtility.StridesTypeToC(isFixedStrides, buffer.Strides);
        var distributed = buffer.DistributedType == null ? null : KernelUtility.DistributedToC(buffer.DistributedType);
        return new(buffer.Name, elemType, buffer.ElemType.SizeInBytes, rank, offset, size, isFixedDims, isFixedStrides, buffer.Dimensions.ToArray(), dims, buffer.Strides.ToArray(), strides, distributed);
    }
}

public record NTTTargetOptionsModel(NTTTargetOptions Options, ulong Alignment, ulong CollectivePoolSize)
{
}

public static class CSourceBuiltn
{
    public const string KernelHeader = @"#pragma once
#include <nncase/ntt/ntt.h>
#include <nncase/ntt/runtime.h>
#include ""topo_aware_runtime.h""
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::distributed::shard_policy;

";

    public static string TopoAwareRuntimeDef(NTTTargetOptions options, ulong dataAlign, ulong collective_pool_size)
    {
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/topo_aware_runtime.cshtml", new NTTTargetOptionsModel(options, dataAlign, collective_pool_size)).Result;
        return content;
    }

    public static string TopologyDef(NTTTargetOptions options)
    {
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/topology_def.h.cshtml", options).Result;
        return content;
    }

    public static string CMakeDef(string name)
    {
        var cmakePath = CMakePath(Path.Combine(Path.GetDirectoryName(typeof(CSourceBuiltn).Assembly.Location)!, "Runtime", "cmake", "ntt_module.cmake"));
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/CMakeLists.txt.cshtml", new { CMakePath = cmakePath }).Result;
        return content;
    }

    public static string MakeMain(TIR.PrimFunction primFunction, ulong dataAlign, ulong dataUsage, ulong rdataPoolSize, ulong localRdataPoolSize, IEnumerable<TIR.Buffer> rdataBuffers, NTTTargetOptions options)
    {
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/thread_main.cpp.cshtml", new KernelMainModel(primFunction, rdataBuffers.ToArray(), options, dataAlign, dataUsage, rdataPoolSize, localRdataPoolSize)).Result;
        return content;
    }

    public static string MakeKernel(string ctype, string kernelImpl)
    {
        return KernelHeader + ctype + kernelImpl;
    }

    private static string CMakePath(string path) =>
        path.Replace("\\", "/", StringComparison.Ordinal);
}
