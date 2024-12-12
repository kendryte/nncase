// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.CPU;

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct DescHeader
{
    [MarshalAs(UnmanagedType.U4)]
    public uint ThreadDim;

    [MarshalAs(UnmanagedType.U4)]
    public uint BlockDim;

    [MarshalAs(UnmanagedType.U4)]
    public uint ChipDim;

    [MarshalAs(UnmanagedType.U4)]
    public uint Reserved0;
}

internal sealed class LinkedModule : ILinkedModule
{
    public const string KernelHeaderSectionName = ".desc";

    public unsafe LinkedModule(IReadOnlyList<ILinkedFunction> functions, Stream desc, Stream text, Stream rdata, IReadOnlyList<Stream> localRdatas, ulong rdataAlign)
    {
        Functions = functions;
        Sections =
        [
            new LinkedSection(desc, KernelHeaderSectionName, 0, 8, (uint)sizeof(DescHeader)),
            new LinkedSection(text, WellknownSectionNames.Text, 0, 8, (ulong)text.Length),
            new LinkedSection(rdata, WellknownSectionNames.Rdata, 0, (uint)rdataAlign, (ulong)rdata.Length),
            new LinkedMultipleContentsSection(localRdatas, WellknownSectionNames.LocalRdata, 0, (uint)rdataAlign),
        ];
    }

    public string ModuleKind => "cpu";

    public uint Version => 0;

    public IReadOnlyList<ILinkedFunction> Functions { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
