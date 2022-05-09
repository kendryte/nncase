// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.CodeGen;

internal class LinkedSection : ILinkedSection
{
    private readonly byte[]? _content;

    public LinkedSection(byte[]? content, string name, int attributes, int alignment, int sizeInMemory)
    {
        if (alignment == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alignment));
        }

        SizeInFile = content?.Length ?? 0;
        if (sizeInMemory < SizeInFile)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInMemory));
        }

        _content = content;
        Name = name;
        Attributes = attributes;
        Alignment = alignment;
        SizeInMemory = sizeInMemory;
    }

    public string Name { get; }

    public int Attributes { get; }

    public int Alignment { get; }

    public int SizeInFile { get; }

    public int SizeInMemory { get; }

    public void Serialize(Stream output)
    {
        if (_content != null)
        {
            output.Write(_content);
        }
    }
}
