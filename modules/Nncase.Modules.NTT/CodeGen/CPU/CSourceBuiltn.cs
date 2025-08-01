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

public record KernelMainModel(TIR.PrimFunction PrimFunction, NTTTargetOptions Options, ulong Alignment, ulong DataSize, ulong RDataSize, ulong ThreadLocalRdataPoolSize, ulong BlockLocalRdataPoolSize)
{
    public BufferRenderInfo GetInfo(TIR.Buffer buffer)
    {
        ulong offset = ((TensorConst)buffer.MemSpan.Buffer.Start).Value.ToScalar<ulong>() + (ulong)buffer.MemSpan.Start.FixedValue;

        var elemType = buffer.ElemType.ToC();
        var rank = buffer.Dimensions.Length;
        var size = buffer.MemSpan.Size / buffer.ElemType.SizeInBytes;
        var isFixedDims = buffer.Dimensions.AsValueEnumerable().All(d => d.IsFixed);
        var isFixedStrides = buffer.Strides.AsValueEnumerable().All(d => d.IsFixed);
        var dims = KernelUtility.DimensionsTypeToC(isFixedDims, buffer.Dimensions);
        var strides = KernelUtility.StridesTypeToC(isFixedStrides, buffer.Strides);
        var distributed = buffer.DistributedType == null ? null : KernelUtility.ShardingToC(buffer.DistributedType);
        return new(buffer.Name, elemType, buffer.ElemType.SizeInBytes, rank, offset, size, isFixedDims, isFixedStrides, buffer.Dimensions.ToArray(), dims, buffer.Strides.ToArray(), strides, distributed);
    }
}

public record NTTTargetOptionsModel(NTTTargetOptions Options, ulong Alignment, ulong CollectivePoolSize)
{
}

public static class CSourceBuiltn
{
    public const string DeviceHeader = @"#pragma once
#include <nncase/ntt/ntt.h>
#include ""topo_aware_runtime.h""
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::distributed::shard_policy;

";

    public const string KernelDeclareHeader = @"#pragma once
#include <nncase/ntt/ntt.h>
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::distributed::shard_policy;

";

    public const string KernelHeader = @"#pragma once
#include <nncase/ntt/ntt.h>
#include ""device_functions.h""
#include ""kernel_functions.h""
#include ""topo_aware_runtime.h""
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::distributed::shard_policy;

";

    public const string ThreadMainHeader = @"#include <nncase/ntt/ntt.h>
#include ""kernel_functions.h""
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::distributed::shard_policy;

";

    public static string TopoAwareRuntimeDef(NTTTargetOptions options, ulong dataAlign, ulong collective_pool_size)
    {
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/topo_aware_runtime.cshtml", new NTTTargetOptionsModel(options, dataAlign, collective_pool_size)).Result;
        return content;
    }

    public static string ModuleTopologyDef(NTTTargetOptions options)
    {
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/module_topology_def.h.cshtml", options).Result;
        return content;
    }

    public static string CMakeDef()
    {
        var cmakePath = CMakePath(Path.Combine(Path.GetDirectoryName(typeof(CSourceBuiltn).Assembly.Location)!, "Runtime", "cmake", "ntt_module.cmake"));
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/CMakeLists.txt.cshtml", new { CMakePath = cmakePath }).Result;
        return content;
    }

    public static string MakeMain(TIR.PrimFunction primFunction, ulong dataAlign, ulong dataUsage, ulong rdataPoolSize, ulong threadLocalRdataPoolSize, ulong blockLocalRdataPoolSize, NTTTargetOptions options)
    {
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/thread_main.cpp.cshtml", new KernelMainModel(primFunction, options, dataAlign, dataUsage, rdataPoolSize, threadLocalRdataPoolSize, blockLocalRdataPoolSize)).Result;
        return content;
    }

    public static string MakeKernel(string ctype, string kernelImpl)
    {
        return KernelHeader + ctype + kernelImpl;
    }

    private static string CMakePath(string path) =>
        path.Replace("\\", "/", StringComparison.Ordinal);
}
