// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Extension.Mathematics;

namespace Nncase.CodeGen;

internal sealed class LinkedModel : ILinkedModel
{
    private const int _minAlignmnet = 8;

    public LinkedModel(FunctionId? entry, IReadOnlyList<ILinkedModule> modules)
    {
        Entry = entry;
        Modules = modules;
    }

    public FunctionId? Entry { get; }

    public IReadOnlyList<ILinkedModule> Modules { get; }

    public void Serialize(Stream output)
    {
        using var writer = new BinaryWriter(output, Encoding.UTF8, true);
        var alignment = LCM(from m in Modules
                            from s in m.Sections
                            select (int)s.Alignment);

        var modelHeader = new ModelHeader
        {
            Identifier = ModelInfo.Identifier,
            Version = ModelInfo.Version,
            Flags = 0,
            Alignment = (uint)alignment,
            Modules = (uint)Modules.Count,
            EntryFunction = Entry?.Id ?? ModelInfo.ModelHasNoEntry,
            EntryModule = Entry?.ModuleId ?? ModelInfo.ModelHasNoEntry,
        };
        writer.Write(ref modelHeader);

        foreach (var module in Modules)
        {
            Serialize(writer, module);
        }
    }

    private static unsafe void FillModuleKind(ref ModuleHeader header, string source)
    {
        fixed (byte* kind = header.Kind)
        {
            if (Encoding.UTF8.GetBytes(source, new Span<byte>(kind, ModelInfo.MaxModuleKindLength)) < 1)
            {
                throw new ArgumentException("Invalid module kind");
            }
        }
    }

    private static unsafe void FillSectionName(ref SectionHeader header, string source)
    {
        fixed (byte* kind = header.Name)
        {
            if (Encoding.UTF8.GetBytes(source, new Span<byte>(kind, ModelInfo.MaxSectionNameLength)) < 1)
            {
                throw new ArgumentException("Invalid section name");
            }
        }
    }

    private static int LCM(IEnumerable<int> source)
    {
        return source.Aggregate(_minAlignmnet, (a, b) => Operations.LCM(a, b));
    }

    private unsafe void Serialize(BinaryWriter writer, ILinkedModule module)
    {
        var header = new ModuleHeader
        {
            Version = module.Version,
            Sections = (uint)module.Sections.Count,
            Functions = (uint)module.Functions.Count,
        };
        FillModuleKind(ref header, module.ModuleKind);

        var headerPos = writer.Position();
        writer.Skip((ulong)sizeof(ModuleHeader));
        foreach (var func in module.Functions)
        {
            Serialize(writer, func);
        }

        foreach (var section in module.Sections)
        {
            Serialize(writer, section);
        }

        writer.AlignPosition(_minAlignmnet);
        var endPos = writer.Position();

        // Write header
        header.Size = (uint)(endPos - headerPos);
        writer.Position(headerPos);
        writer.Write(ref header);
        writer.Position(endPos);
    }

    private unsafe void Serialize(BinaryWriter writer, ILinkedFunction func)
    {
        var funcHeader = new FunctionHeader
        {
            Parameters = (uint)func.ParameterTypes.Count,
            Entrypoint = func.TextBegin,
            TextSize = func.TextLength,
            Sections = (uint)func.Sections.Count,
        };

        var headerPos = writer.Position();
        writer.Skip((ulong)sizeof(FunctionHeader));

        foreach (var type in func.ParameterTypes)
        {
            TypeSerializer.Serialize(writer, type);
        }

        TypeSerializer.Serialize(writer, func.ReturnType);

        foreach (var section in func.Sections)
        {
            Serialize(writer, section);
        }

        writer.AlignPosition(_minAlignmnet);
        var endPos = writer.Position();

        // Write header
        funcHeader.Size = (uint)(endPos - headerPos);
        writer.Position(headerPos);
        writer.Write(ref funcHeader);
        writer.Position(endPos);
    }

    private unsafe void Serialize(BinaryWriter writer, ILinkedSection section)
    {
        var header = new SectionHeader
        {
            Flags = section.Flags,
            BodySize = section.SizeInFile,
            MemorySize = section.SizeInMemory,
        };
        FillSectionName(ref header, section.Name);

        var headerPos = writer.Position();
        writer.Skip((ulong)sizeof(SectionHeader));

        header.BodyStart = (uint)writer.AlignPosition(section.Alignment);
        writer.Flush();
        section.Serialize(writer.BaseStream);

        writer.AlignPosition(_minAlignmnet);
        var endPos = writer.Position();

        // Write header
        header.Size = (uint)(endPos - headerPos);
        writer.Position(headerPos);
        writer.Write(ref header);
        writer.Position(endPos);
    }
}
