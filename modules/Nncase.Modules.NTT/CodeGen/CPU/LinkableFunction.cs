// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Nncase.IR;

namespace Nncase.CodeGen.NTT;

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct KernelDescHeader
{
    [MarshalAs(UnmanagedType.U4)]
    public uint OutputAlign;

    [MarshalAs(UnmanagedType.U4)]
    public uint Reserved0;

    [MarshalAs(UnmanagedType.U8)]
    public ulong OutputPoolSize;
}

internal sealed class LinkableKernelFunction : ILinkableFunction
{
    public const string KernelHeaderSectionName = ".desc";

    public LinkableKernelFunction(uint id, TIR.PrimFunction sourceFunction, KernelCSource funcCSource, Stream text, params ILinkedSection[] sections)
    {
        Id = id;
        SourceFunction = sourceFunction;
        PrimFunction = sourceFunction;
        FunctionCSource = funcCSource;
        Text = text;
        Sections = sections;
    }

    public uint Id { get; }

    public BaseFunction SourceFunction { get; }

    public TIR.PrimFunction PrimFunction { get; }

    public KernelCSource FunctionCSource { get; }

    public Stream Text { get; }

    public IEnumerable<FunctionRef> FunctionRefs => Enumerable.Empty<FunctionRef>();

    public IReadOnlyList<ILinkedSection> Sections { get; }
}

internal sealed class LinkableDeviceFunction : ILinkableFunction
{
    public LinkableDeviceFunction(uint id, TIR.PrimFunction sourceFunction, string header, Stream text)
    {
        Id = id;
        SourceFunction = sourceFunction;
        Header = header;
        PrimFunction = sourceFunction;
        Text = text;
        Sections = Array.Empty<ILinkedSection>();
    }

    public uint Id { get; }

    public BaseFunction SourceFunction { get; }

    public string Header { get; }

    public TIR.PrimFunction PrimFunction { get; }

    public Stream Text { get; }

    public IEnumerable<FunctionRef> FunctionRefs => Enumerable.Empty<FunctionRef>();

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
