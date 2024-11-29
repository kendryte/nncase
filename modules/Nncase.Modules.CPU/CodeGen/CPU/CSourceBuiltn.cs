﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Runtime.CompilerServices;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.Targets;
using Razor.Templating.Core;

namespace Nncase.CodeGen.CPU;

public record BufferRenderInfo(string Name, string ElemType, ulong Offset, ulong Size, string Dimensions, string Strides)
{
}

public record KernelMainModel(TIR.PrimFunction PrimFunction, TIR.Buffer[] RDataBuffers, CpuTargetOptions Options, ulong Alignment, ulong DataSize, ulong RDataSize)
{
    public BufferRenderInfo GetInfo(TIR.Buffer buffer)
    {
        ulong offset = 0;
        if (buffer.MemSpan.Start is IR.TensorConst tc)
        {
            offset = tc.Value.Cast<ulong>()[0];
        }

        var elemType = buffer.ElemType.ToC();
        var size = ((IR.TensorConst)buffer.MemSpan.Size).Value.Cast<ulong>()[0] / (ulong)buffer.ElemType.SizeInBytes;
        var dims = KernelUtility.DimensionsToC(buffer.Dimensions);
        var strides = KernelUtility.StridesToC(buffer.Strides);
        return new(buffer.Name, elemType, offset, size, dims, strides);
    }
}

public record CpuTargetOptionsModel(CpuTargetOptions Options, ulong Alignment, ulong CollectivePoolSize)
{
}

public static class CSourceBuiltn
{
    public const string KernelHeader = @"#pragma once
#include <nncase/ntt/ntt.h>
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::distributed::shard_policy;

";

    public static string TopoAwareRuntimeDef(CpuTargetOptions options, ulong dataAlign, ulong collective_pool_size)
    {
        if (options.Hierarchies[0].Any(i => i > 1))
        {
            var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/topo_aware_runtime.cshtml", new CpuTargetOptionsModel(options, dataAlign, collective_pool_size)).Result;
            return content;
        }

        return string.Empty;
    }

    public static string TopologyDef(CpuTargetOptions options)
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

    public static string MakeMain(TIR.PrimFunction primFunction, ulong dataAlign, ulong dataUsage, ulong rdataPoolSize, IEnumerable<TIR.Buffer> rdataBuffers, CpuTargetOptions options)
    {
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/thread_main.cpp.cshtml", new KernelMainModel(primFunction, rdataBuffers.ToArray(), options, dataAlign, dataUsage, rdataPoolSize)).Result;
        return content;
    }

    public static string MakeKernel(string ctype, string kernelImpl)
    {
        return KernelHeader + ctype + kernelImpl;
    }

    private static string CMakePath(string path) =>
        path.Replace("\\", "/", StringComparison.Ordinal);
}
