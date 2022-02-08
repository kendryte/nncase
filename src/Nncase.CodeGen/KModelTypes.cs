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
    public uint HeaderSize;
    public uint Flags;
    public uint Alignment;
    public uint Modules;
    public uint EntryModule;
    public uint EntryFunction;
}

;

[StructLayout(LayoutKind.Sequential)]
public struct FunctionHeader
{
    public uint HeaderSize;
    public uint Size;
    public uint InputPoolSize;
    public uint OutputPoolSize;
    public uint Inputs;
    public uint Outputs;
    public uint Entrypoint;
    public uint TextSize;
}

;

[StructLayout(LayoutKind.Sequential)]
public struct ModuleHeader
{
    public ModuleType Type;
    public uint Version;
    public uint HeaderSize;
    public uint Size;
    public uint Mempools;
    public uint SharedMempools;
    public uint Sections;
    public uint Functions;
    public uint Reserved0;
}

;

[StructLayout(LayoutKind.Sequential)]
public struct MemPoolDesc
{
    public Schedule.MemoryLocation Location;
    public byte[] Reserved0 = new byte[3];
    public uint Size;
}

;

[StructLayout(LayoutKind.Sequential)]
public struct SharedMempoolDesc
{
    public uint Module;
    public uint Size;
}

;

[StructLayout(LayoutKind.Sequential)]
public struct SectionHeader
{
    public char[] Name = new char[ModelInfo.MaxSectionNameLength];
    public uint Flags;
    public uint BodyStart;
    public uint BodySize;
    public uint Reserved0;
}

;

[StructLayout(LayoutKind.Sequential)]
public struct shape_header
{
    public uint Size;

    //     const uint* begin() const noexcept
    //     {
    //     return reinterpret_cast<const uint*>(reinterpret_cast<uintptr_t>(this) + sizeof(shape_header));
    // }

    // const uint* end() const noexcept
    // {
    //     return begin() + size;
    // }

    //         uint operator[](size_t index) const
    //             {
    //         assert(index<size);
    // return begin()[index];
    //     }
}
