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

public enum FunctionIdComponent
{
    ModuleId,
    FunctionId,
}

public static class WellknownSectionNames
{
    public static readonly string Text = ".text";

    public static readonly string Rdata = ".rdata";
}

public class Symbol
{
    public Symbol(string section, long position)
    {
        Section = section;
        Position = position;
    }

    public string Section { get; }

    public long Position { get; }
}

public record SymbolRef(long Position, int Length, Symbol Symbol, bool Relative, int Offset);

public record FunctionRef(long Position, int Length, BaseFunction Callable, FunctionIdComponent Component, int Offset);

public class SectionManager
{
    private readonly Dictionary<string, (MemoryStream Stream, BinaryWriter Writer)> _sections = new();

    public BinaryWriter GetWriter(string name)
    {
        if (!_sections.TryGetValue(name, out var section))
        {
            var stream = new MemoryStream();
            section = (stream, new BinaryWriter(stream, Encoding.UTF8, true));
            _sections.Add(name, section);
        }

        return section.Writer;
    }

    public byte[]? GetContent(string name)
    {
        if (_sections.TryGetValue(name, out var section))
        {
            section.Writer.Flush();
            return section.Stream.ToArray();
        }

        return null;
    }
}
