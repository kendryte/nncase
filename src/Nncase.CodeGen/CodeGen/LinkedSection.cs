// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.CodeGen;

public class LinkedSection : ILinkedSection
{
    private readonly byte[]? _content;

    public LinkedSection(byte[]? content, string name, uint flags, uint alignment, uint sizeInMemory)
    {
        if (alignment == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alignment));
        }

        SizeInFile = (uint?)content?.Length ?? 0;
        if (sizeInMemory < SizeInFile)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInMemory));
        }

        _content = content;
        Name = name;
        Flags = flags;
        Alignment = alignment;
        SizeInMemory = sizeInMemory;
    }

    public string Name { get; }

    public uint Flags { get; }

    public uint Alignment { get; }

    public uint SizeInFile { get; }

    public uint SizeInMemory { get; }

    public void Serialize(Stream output)
    {
        if (_content != null)
        {
            output.Write(_content);
        }
    }
}
