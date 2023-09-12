﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.CodeGen;

public class LinkedSection : ILinkedSection
{
    private readonly Stream? _content;

    public LinkedSection(Stream? content, string name, uint flags, uint alignment, ulong sizeInMemory)
    {
        if (alignment == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alignment));
        }

        SizeInFile = (ulong?)content?.Length ?? 0;
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

    public ulong SizeInFile { get; }

    public ulong SizeInMemory { get; }

    public void Serialize(Stream output)
    {
        if (_content != null)
        {
            _content.Seek(0, SeekOrigin.Begin);
            _content.CopyTo(output);
        }
    }
}
