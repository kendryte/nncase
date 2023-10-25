// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Nncase.IR;

namespace Nncase.CodeGen.XPU;

internal sealed class LinkableFunction : ILinkableFunction
{
    public LinkableFunction(uint id, TIR.PrimFunction sourceFunction, FunctionCSource funcCSource, Stream text)
    {
        Id = id;
        SourceFunction = sourceFunction;
        PrimFunction = sourceFunction;
        FunctionCSource = funcCSource;
        Text = text;
        Sections = Array.Empty<ILinkedSection>();
    }

    public uint Id { get; }

    public BaseFunction SourceFunction { get; }

    public TIR.PrimFunction PrimFunction { get; }

    public FunctionCSource FunctionCSource { get; }

    public Stream Text { get; }

    public IEnumerable<FunctionRef> FunctionRefs => Enumerable.Empty<FunctionRef>();

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
