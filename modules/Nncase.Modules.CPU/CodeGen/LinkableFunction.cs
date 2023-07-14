// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.CodeGen.CPU;

internal sealed class LinkableFunction : ILinkableFunction
{
    private readonly byte[] _desc;

    public LinkableFunction(uint id, TIR.PrimFunction sourceFunction, byte[] text, byte[] desc)
    {
        Id = id;
        SourceFunction = sourceFunction;
        Text = text;
        _desc = desc;
        Sections = new LinkedSection[] { new(_desc, ".desc", 0, 8, (uint)_desc.Length) };
    }

    public uint Id { get; }

    public BaseFunction SourceFunction { get; }

    public byte[] Text { get; }

    public IEnumerable<FunctionRef> FunctionRefs => Enumerable.Empty<FunctionRef>();

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
