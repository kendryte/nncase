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

[StructLayout(LayoutKind.Sequential)]
public struct ModelHeader
{
    public uint Identifier;
    public uint Version;
    public uint Flags;
    public uint Alignment;
    public uint Modules;
    public uint EntryModule;
    public uint EntryFunction;
    public uint Reserved0;
}

[StructLayout(LayoutKind.Sequential)]
public struct FunctionHeader
{
    public uint Parameters;
    public uint Entrypoint;
    public uint TextSize;
    public uint Size;
    public uint Sections;
    public uint Reserved0;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct ModuleHeader
{
    public fixed byte Kind[ModelInfo.MaxModuleKindLength];
    public uint Version;
    public uint Size;
    public uint Sections;
    public uint Functions;
}

/// <summary>
/// the section header.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct SectionHeader
{
    public fixed byte Name[ModelInfo.MaxSectionNameLength];
    public uint Flags;
    public uint Size;
    public uint BodyStart;
    public uint BodySize;
    public uint MemorySize;
    public uint Reserved0;
}

[StructLayout(LayoutKind.Sequential)]
public struct Shape_header
{
    public uint Size;

    // const uint* begin() const noexcept
    //     {
    //     return reinterpret_cast<const uint*>(reinterpret_cast<uintptr_t>(this) + sizeof(shape_header));
    // }

    // const uint* end() const noexcept
    // {
    //     return begin() + size;
    // }

    // uint operator[](size_t index) const
    //             {
    //         assert(index<size);
    // return begin()[index];
    //     }
}
