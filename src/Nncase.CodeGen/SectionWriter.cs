// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.CodeGen;

public struct SymbolRef
{
    public string Name;
    public long Streampos;
    public ulong Bitoffset;
    public ulong Length;
}

public struct Symbol
{
    public string Name;
    public long Streampos;
}

public class SectionWriter : BinaryWriter
{
    public SectionWriter(Stream output) : base(output) { }

    public IReadOnlyList<Symbol> Symbols => symbols_;

    public IReadOnlyList<SymbolRef> SymbolRefs => symbol_refs_;

    readonly List<SymbolRef> symbol_refs_ = new();
    readonly List<Symbol> symbols_ = new();

    public void AddSymbolRef(ulong offset, ulong length, string name)
    {
        symbol_refs_.Add(new() { Name = name, Streampos = BaseStream.Position, Bitoffset = offset, Length = length });
    }

    public void AddSymbol(string name)
    {
        symbols_.Add(new() { Name = name, Streampos = BaseStream.Position });
    }
}