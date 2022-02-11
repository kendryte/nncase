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
/// The CodeGen Model Information.
/// </summary>
public static class ModelInfo
{
    /// <summary>
    /// Gets model identifier.
    /// </summary>
    public static class ModelInfo
    {
        public static uint IDENTIFIER => BitConverter.ToUInt32(Encoding.UTF8.GetBytes("LDMK"), 0);
        public const uint VERSION = 5;
        public const uint SECTION_MERGED_INTO_RDATA = 1;
        public const uint MAX_SECTION_NAME_LENGTH = 16;
        public const uint MAX_MODULE_TYPE_LENGTH = 16;
    }

    /// <summary>
    /// create the modult type by name.
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public static ModuleType Create(string name)
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = (int)ModelInfo.MAX_MODULE_TYPE_LENGTH)]
        char[] Types = new char[ModelInfo.MAX_MODULE_TYPE_LENGTH];

        /// <summary>
        /// create the modult type by name
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public static ModuleType Create(string name)
        {
            mt.Types[i] = i < name.Length ? name[i] : '\0';
        }

        return mt;
    }
}

/// <summary>
/// the runtime function.
/// </summary>
public interface IRTFunction
{
    /// <summary>
    /// the funtion name.
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// the function pointer handle.
    /// </summary>
    public Delegate Handle { get; set; }
}

/// <summary>
/// when model or module has been Serialized. return the result.
/// </summary>
public interface ISerializeResult
{
}

/// <summary>
/// runtime module.
/// </summary>
public interface IRTModule
{
    /// <summary>
    /// the Module type.
    /// </summary>
    public ModuleType ModuleType { get; set; }

    /// <summary>
    /// get source code.
    /// </summary>
    public interface IRTModule
    {
        /// <summary>
        /// the Module type
        /// </summary>
        public ModuleType ModuleType { get; }

        /// <summary>
        /// get source code
        /// </summary>
        public byte[] Source { get; }

        /// <summary>
        /// get source file ext
        /// </summary>
        public string SourceExt { get; }

        /// <summary>
        /// is runtime module has been Serialized?
        /// </summary>
        public bool IsSerialized { get; }

        /// <summary>
        /// dump the source into file `DumpDirPath/name.xx`
        /// </summary>
        /// <param name="name">file name. </param>
        /// <param name="dumpDirPath"> dump dir path. </param>
        public void Dump(string name, string dumpDirPath);

        /// <summary>
        /// Serialize the code
        /// </summary>
        public ISerializeResult Serialize();

        /// <summary>
        /// get the all runtime function
        /// </summary>
        public IReadOnlyList<IRTFunction> Functions { get; }
    }

    /// <summary>
    /// get the all runtime function.
    /// </summary>
    public IReadOnlyList<IRTFunction> Functions { get; }
}

/// <summary>
/// The runtime Model.
/// <example>
/// Runtime Model ∈ {
///     RTModule1 ∈ { RTFuntion1, RTFuntion2 ... } ,
///     RTModule2 ...}
/// </example>
/// </summary>
public interface IRTModel
{
    /// <summary>
    /// the target.
    /// </summary>
    public ITarget Target { get; set; }

    /// <summary>
    /// schedule result.
    /// </summary>
    public Schedule.SchedModelResult modelResult { get; set; }

    /// <summary>
    /// get source code.
    /// </summary>
    public string Source { get; set; }

    /// <summary>
    /// get source file ext.
    /// </summary>
    public string SourceExt { get; set; }

    /// <summary>
    /// dump the source into file `DumpDirPath/name.xx`.
    /// </summary>
    /// <param name="name">file name. </param>
    /// <param name="dumpDirPath"> dump dir path. </param>
    public void Dump(string name, string dumpDirPath);

    /// <summary>
    /// Serialize the runtime model.
    /// <example>
    /// if this runtime model is kmodel, will serialize as kmodel.
    /// if is Csource, will serialize as .c file.
    /// </example>
    /// </summary>
    public interface IRTModel
    {
        /// <summary>
        /// the target
        /// </summary>
        public ITarget Target { get; set; }

        /// <summary>
        /// schedule result
        /// </summary>
        public Schedule.SchedModelResult modelResult { get; set; }

        /// <summary>
        /// get source code
        /// </summary>
        public byte[] Source { get; }

        /// <summary>
        /// get source file ext
        /// </summary>
        public string SourceExt { get; }

        /// <summary>
        /// the source file path
        /// </summary>
        public string SourcePath { get; }

        /// <summary>
        /// this rt model is Serialized?
        /// </summary>
        public bool IsSerialized { get; }

        /// <summary>
        /// dump the source into file `DumpDirPath/name.xx`
        /// </summary>
        /// <param name="name">file name. </param>
        /// <param name="dumpDirPath"> dump dir path. </param>
        /// <returns> dumped path. </returns>
        public string Dump(string name, string dumpDirPath);

        /// <summary>
        /// Serialize the runtime model
        /// <example>
        /// if this runtime model is kmodel, will serialize as kmodel.
        /// if is Csource, will serialize as .c file.
        /// </example>
        /// </summary>
        public ISerializeResult Serialize();

        /// <summary>
        /// call this runtime model
        /// </summary>
        /// <param name="args"> input args</param>
        /// <returns></returns>
        public object? Invoke(params object?[]? args);

        /// <summary>
        /// get runtime function entry
        /// </summary>
        public IRTFunction? Entry { get; }

        /// <summary>
        /// get the all runtime function
        /// </summary>
        public IReadOnlyList<IRTModule> Modules { get; }
    }
}
