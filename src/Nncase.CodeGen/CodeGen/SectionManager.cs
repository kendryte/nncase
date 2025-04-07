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

public static class WellknownSectionNames
{
    public static readonly string Text = ".text";

    public static readonly string Rdata = ".rdata";

    public static readonly string LocalRdata = ".local_rdata";
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

public class SectionManager
{
    private readonly Dictionary<(string Name, int ShardId), (Stream Stream, BinaryWriter Writer)> _sections = new();

    public BinaryWriter GetWriter(string name, int shardId = 0)
    {
        if (!_sections.TryGetValue((name, shardId), out var section))
        {
            var tmpFile = File.Open(Path.GetTempFileName(), new FileStreamOptions
            {
                Access = FileAccess.ReadWrite,
                Mode = FileMode.Create,
                Options = FileOptions.Asynchronous | FileOptions.DeleteOnClose,
            });
            section = (tmpFile, new BinaryWriter(tmpFile, Encoding.UTF8, true));
            _sections.Add((name, shardId), section);
        }

        return section.Writer;
    }

    public Stream? GetContent(string name, int shardId = 0)
    {
        if (_sections.TryGetValue((name, shardId), out var section))
        {
            section.Writer.Flush();
            return section.Stream;
        }

        return null;
    }
}
