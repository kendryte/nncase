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
/// Gets model identifier.
/// </summary>
public static class ModelInfo
{
    public static readonly uint ModelHasNoEntry = unchecked((uint)-1);

    /// <summary>
    /// the idenitifer.
    /// </summary>
    public static readonly uint IDENTIFIER = BitConverter.ToUInt32(Encoding.UTF8.GetBytes("LDMK"), 0);

    /// <summary>
    /// kmodel version.
    /// </summary>
    public const int VERSION = 6;
    /// <summary>
    /// merged rdata flag.
    /// </summary>
    public const int SECTION_MERGED_INTO_RDATA = 1;
    /// <summary>
    /// max section name length.
    /// </summary>
    public const int MAX_SECTION_NAME_LENGTH = 16;
    /// <summary>
    /// max module type length.
    /// </summary>
    public const int MAX_MODULE_KIND_LENGTH = 16;
}

/// <summary>
/// the module type
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct ModuleType
{
    /// <summary>
    /// the module types
    /// </summary>
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = (int)ModelInfo.MAX_MODULE_KIND_LENGTH)]
    public string Types;

    /// <summary>
    /// create the modult type by name
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public static ModuleType Create(string name)
    {
        var mt = new ModuleType();
        var chars = new char[ModelInfo.MAX_MODULE_KIND_LENGTH];
        for (int i = 0; i < ModelInfo.MAX_MODULE_KIND_LENGTH; i++)
        {
            chars[i] = i < name.Length ? name[i] : '\0';
        }

        mt.Types = new(chars);
        return mt;
    }
}
