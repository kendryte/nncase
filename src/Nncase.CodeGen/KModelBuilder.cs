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

/// <summary>
/// the kmodule Serialized result.
/// </summary>
public class KModuleSerializeResult : ISerializeResult
{
    public uint Alignment { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="KModuleSerializeResult"/> class.
    /// </summary>
    /// <param name="alignment"></param>
    public KModuleSerializeResult(uint alignment)
    {
        Alignment = alignment;
    }
}

/// <summary>
/// the kmodel Serialized result.
/// </summary>
public class KModelSerializeResult : ISerializeResult
{
    public int ModelSize;
    public KModelSerializeResult(int size)
    {
        ModelSize = size;
    }
}

/// <summary>
/// the kmodel format runtime model.
/// </summary>
public class RTKModel : IRTModel
{
    List<IRTModule> modules;
    string sourcePath;
    bool isSerialized;
    KModelSerializeResult serializeResult;

    /// <summary>
    /// create the kmodel.
    /// </summary>
    /// <param name="result"></param>
    /// <param name="target"></param>
    public RTKModel(Schedule.SchedModelResult result, ITarget target)
    {
        Target = target;
        modelResult = result;
        Source = string.Empty;
        modules = new();
        sourcePath = CodeGenUtil.GetTempFileName(SourceExt);
        isSerialized = false;
        serializeResult = new(0);
    }

    /// <inheritdoc/>
    public ITarget Target { get; set; }

    /// <inheritdoc/>
    public Schedule.SchedModelResult modelResult { get; set; }

    /// <inheritdoc/>
    public string Source { get; set; }

    /// <inheritdoc/>
    public string SourceExt { get => "kmodel"; set { } }

    /// <inheritdoc/>
    public IRTFunction? Entry { get; set; }

    /// <inheritdoc/>
    public IReadOnlyList<IRTModule> Modules => modules;

    /// <inheritdoc/>
    public void Dump(string name, string dumpDirPath)
    {
        if (isSerialized)
        {
            File.Copy(sourcePath, Path.Combine(dumpDirPath, $"{name}.{SourceExt}"));
        }
    }

    /// <inheritdoc/>
    public object? Invoke(params object?[]? args)
    {
        throw new NotImplementedException();
    }

    /// <inheritdoc/>
    public ISerializeResult Serialize()
    {
        if (isSerialized)
        {
            return serializeResult;
        }

        // step 1. create the kmodel file
        using var ostream = File.Open(sourcePath, FileMode.OpenOrCreate, FileAccess.ReadWrite);
        using var writer = new BinaryWriter(ostream);
        var begin_pos = writer.BaseStream.Position;

        // step 2. start write.
        var header = new ModelHeader()
        {
            Identifier = ModelInfo.Identifier,
            Version = ModelInfo.Version,
            HeaderSize = (uint)Marshal.SizeOf(typeof(ModelHeader)),
            Flags = 0,
            Alignment = 8,
            Modules = (uint)modelResult.Modules.Count,
        };

        var header_pos = writer.BaseStream.Position;
        writer.Seek(Marshal.SizeOf(typeof(ModelHeader)), SeekOrigin.Current);
        foreach (var moduleRes in modelResult.Modules)
        {
            var rtmodule = Target.CreateRTModule(moduleRes.ModuleType, moduleRes, modelResult);
            var res = (KModuleSerializeResult)rtmodule.Serialize();
            modules.Add(rtmodule);
            writer.Write(rtmodule.Source);
            header.Alignment = Math.Max(header.Alignment, res.Alignment);
        }

        // Entry point
        for (int i = 0; i < modelResult.Modules.Count; i++)
        {
            var mod_sched = modelResult.Modules[i];
            for (int j = 0; j < mod_sched.Functions.Count; j++)
            {
                if (object.ReferenceEquals(modelResult.Entry, mod_sched.Functions[j]))
                {
                    header.EntryModule = (uint)i;
                    header.EntryFunction = (uint)j;
                }
            }
        }

        var end_pos = writer.BaseStream.Position;

        // write header
        writer.Seek((int)header_pos, SeekOrigin.Begin);
        writer.Write(CodeGenUtil.StructToBytes(header));
        writer.Seek((int)end_pos, SeekOrigin.Begin);
        writer.Flush();
        writer.Close();
        serializeResult.ModelSize = ((int)(end_pos - begin_pos));
        return serializeResult;
    }
}