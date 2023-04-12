// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IR;

namespace Nncase.CodeGen;

/// <summary>
/// the module type.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct ModuleType
{
    /// <summary>
    /// the module types.
    /// </summary>
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = (int)ModelInfo.MaxModuleKindLength)]
    public string Types;

    /// <summary>
    /// create the modult type by name.
    /// </summary>
    public static ModuleType Create(string name)
    {
        var mt = default(ModuleType);
        var chars = new char[ModelInfo.MaxModuleKindLength];
        for (int i = 0; i < ModelInfo.MaxModuleKindLength; i++)
        {
            chars[i] = i < name.Length ? name[i] : '\0';
        }

        mt.Types = new(chars);
        return mt;
    }
}

/// <summary>
/// Gets model identifier.
/// </summary>
public static class ModelInfo
{
    /// <summary>
    /// kmodel version.
    /// </summary>
    public const int Version = 6;

    /// <summary>
    /// merged rdata flag.
    /// </summary>
    public const int SectionMergedIntoRdata = 1;

    /// <summary>
    /// max section name length.
    /// </summary>
    public const int MaxSectionNameLength = 16;

    /// <summary>
    /// max module type length.
    /// </summary>
    public const int MaxModuleKindLength = 16;

    public static readonly uint ModelHasNoEntry = unchecked((uint)-1);

    /// <summary>
    /// the idenitifer.
    /// </summary>
    public static readonly uint Identifier = BitConverter.ToUInt32(Encoding.UTF8.GetBytes("LDMK"), 0);
}
