// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.CodeGen.CPU;

internal sealed class LinkableFunction : ILinkableFunction
{
    public LinkableFunction(uint id, TIR.PrimFunction sourceFunction, FunctionCSource funcCSource)
    {
        Id = id;
        SourceFunction = sourceFunction;
        PrimFunction = sourceFunction;
        FunctionCSource = funcCSource;
        Text = Array.Empty<byte>();
        var desc = System.Text.Encoding.ASCII.GetBytes(sourceFunction.Name);
        Sections = new LinkedSection[] { new(desc, ".desc", 0, 8, (uint)desc.Length) };
    }

    public uint Id { get; }

    public BaseFunction SourceFunction { get; }

    public TIR.PrimFunction PrimFunction { get; }

    public FunctionCSource FunctionCSource { get; }

    public byte[] Text { get; }

    public IEnumerable<FunctionRef> FunctionRefs => Enumerable.Empty<FunctionRef>();

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
