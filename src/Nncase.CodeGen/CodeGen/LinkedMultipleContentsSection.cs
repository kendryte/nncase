// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Utilities;

namespace Nncase.CodeGen;

public class LinkedMultipleContentsSection : ILinkedSection
{
    private readonly IReadOnlyList<Stream> _contents;
    private readonly uint _headerSize;

    public LinkedMultipleContentsSection(IReadOnlyList<Stream> contents, string name, uint flags, uint alignment)
    {
        if (alignment == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alignment));
        }

        _headerSize = (uint)(sizeof(ulong) * 2 * contents.Count);
        _contents = contents;
        Name = name;
        Flags = flags;
        Alignment = alignment;
    }

    public string Name { get; }

    public uint Flags { get; }

    public uint Alignment { get; }

    public ulong SizeInFile { get; private set; }

    public ulong SizeInMemory { get; private set; }

    public void Serialize(Stream output)
    {
        using var writer = new BinaryWriter(output, Encoding.UTF8, leaveOpen: true);
        var contentPositions = new List<(ulong Position, ulong Length)>();

        // 1. Skip header
        var headerPos = writer.Position();
        var contentBegin = writer.Position(headerPos + _headerSize);

        // 2. Write contents
        foreach (var content in _contents)
        {
            writer.AlignPosition(Alignment);
            var offset = writer.Position() - contentBegin;
            content.Seek(0, SeekOrigin.Begin);
            content.CopyTo(output);
            contentPositions.Add(((ulong)offset, (ulong)content.Length));
        }

        // 3. Write header
        var endPos = writer.Position();
        writer.Position(headerPos);
        foreach (var (pos, length) in contentPositions)
        {
            writer.Write(pos);
            writer.Write(length);
        }

        writer.Position(endPos);
        SizeInFile = SizeInMemory = (ulong)(endPos - headerPos);
    }
}
