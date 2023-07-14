// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#if false
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR;

namespace Nncase.CodeGen;

/// <summary>
/// the c source runtime function.
/// </summary>
/// <param name="name"></param>
/// <param name="handle"></param>
public record CSourceRTFunction(string name, Delegate handle) : IRTFunction
{
    public string Name { get => name; set { } }
    public Delegate Handle { get => handle; set { } }
}

public class CSourceSerializeResult : ISerializeResult
{

}

/// <summary>
/// c runtime module impl
/// </summary>
public class CSourceRTModel : IRTModule, IRTModel
{
    /// <inheritdoc/>
    public ModuleType ModuleType { get => CodeGen.ModuleType.Create("CSource"); set { } }

    /// <inheritdoc/>
    public ITarget Target { get; set; }

    /// <inheritdoc/>
    public IReadOnlyList<IRTModule> Modules => throw new NotImplementedException();

    /// <inheritdoc/>
    public string SourcePath { get; private set; }

    public IRModel Model { get; set; }
    IRTFunction? _entry = null;

    /// <inheritdoc/>
    public bool IsSerialized { get; private set; }

    readonly List<IRTFunction> _functions = new();

    /// <summary>
    /// <see cref="CSourceRTModel"/>
    /// </summary>
    public CSourceRTModel(IRModel model, ITarget target)
    {
        SourcePath = CodeGenUtil.GetTempFileName("c");
        Model = model;
        Target = target;
    }

    /// <inheritdoc/>
    public byte[] Source { get => File.ReadAllBytes(SourcePath); set { } }

    /// <inheritdoc/>
    public string SourceExt { get => "c"; set { } }

    /// <inheritdoc/>
    public IRTFunction? Entry => _entry;

    /// <inheritdoc/>
    public IReadOnlyList<IRTFunction> Functions => _functions;

    /// <inheritdoc/>
    string _dllPath = "";

    /// <summary>
    /// write the c source code into source path.
    /// </summary>
    /// <exception cref="InvalidProgramException"></exception>
    void BuildCode()
    {
        if (File.Exists(SourcePath))
            File.Delete(SourcePath);
        using (var writer = new StreamWriter(SourcePath, false, Encoding.UTF8))
        {
            var visior = new CSourceHostBuildVisior(writer);
            if (Model.Entry is null) { throw new InvalidProgramException("The Model Entry Is Null!"); }
            if (Model.Entry.CheckedType is null && Model.Entry.InferenceType() == false) { throw new InvalidProgramException("The Model Entry Can't Inference Type!"); }
            visior.Visit(Model.Entry);
        }
    }

    public void CompileCode()
    {
        if (!File.Exists(SourcePath))
            throw new InvalidProgramException("The Source Code Path Is Invalid!");
        var compiler = new CSourceCompiler();
        _dllPath = compiler.Compile(SourcePath);
    }

    /// <summary>
    /// bind each IR.Funtion with C function
    /// </summary>
    /// <exception cref="InvalidProgramException"></exception>
    public void ExportCode()
    {
        if (!File.Exists(_dllPath))
            throw new InvalidProgramException("The DLL Path Is Invalid!");
        var dllPtr = NativeLibrary.Load(_dllPath);
        foreach (var module in Model.Modules)
        {
            foreach (var f in module.Callables)
            {
                var funcType = f.ToDelegateType(Path.GetFileName(_dllPath));
                var funPtr = NativeLibrary.GetExport(dllPtr, f.Name);
                _functions.Add(new CSourceRTFunction(f.Name, funPtr.BindDelegate(funcType)));
                if (f == Model.Entry) { _entry = _functions.Last(); }
            }
        }
    }

    /// <inheritdoc/>
    public ISerializeResult Serialize()
    {
        if (IsSerialized) { return new CSourceSerializeResult(); }
        BuildCode();
        CompileCode();
        ExportCode();
        return new CSourceSerializeResult();
    }

    /// <summary>
    /// invoke the module entry
    /// </summary>
    /// <param name="args">input args</param>
    /// <returns> results </returns>
    /// <exception cref="InvalidOperationException"></exception>
    public object? Invoke(params object?[]? args)
    {
        if (Entry is null)
            throw new InvalidOperationException("This RTModule Have No Entry Function!");
        return Entry.Handle.DynamicInvoke(args);
    }

    public string Dump(string name, string DumpDirPath)
    {
        var dump_path = $"{DumpDirPath}/{name}.{SourceExt}";
        using var file = File.Open(dump_path, FileMode.OpenOrCreate, FileAccess.Write);
        using var writer = new StreamWriter(file);
        writer.Write(Source);
        return dump_path;
    }

}

/// <summary>
/// the csource code compiler.
/// </summary>
public class CSourceCompiler
{
    /// <summary>
    /// compiler exe name
    /// </summary>
    string _exe = "", _arch = "", _ext = "";

    /// <summary>
    /// select current pattern's exe
    /// </summary>
    /// <exception cref="NotSupportedException"></exception>
    void PlatformSpecific()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            _exe = "gcc";
            _ext = "so";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            _exe = "clang";
            _ext = "dylib";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            _exe = "cmd";
            _ext = "dll";
        }
    }

    void ArchSpecific()
    {
        _arch = RuntimeInformation.OSArchitecture switch
        {
            Architecture.X64 => RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "x86-64" : "x86_64",
            Architecture.Arm64 => "arm64",
            _ => throw new NotSupportedException(RuntimeInformation.OSArchitecture.ToString()),
        };
    }

    string ArgumentsSpecific(string sourcePath, string outPath)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return $"{sourcePath} -fPIC -shared -march={Arch} -o {outPath}";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return $"{sourcePath} -fPIC -shared -arch {Arch} -o {outPath}";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var vsdir = Environment.GetEnvironmentVariable("VSAPPIDDIR") ?? throw new InvalidOperationException("Cannot find vs");
            var vcvardir = Path.Combine(vsdir, "..\\..\\VC\\Auxiliary\\Build\\vcvarsall.bat");
            return $"/C (\"{vcvardir}\" x64) && (cl /D_USRDLL /D_WINDLL \"{sourcePath}\" /MT /link /DLL /OUT:\"{outPath}\")";
        }
        throw new System.ArgumentOutOfRangeException("Only Support Linux/Osx/Windows");
    }

    protected string Exe
    {
        get => _exe;
    }

    protected string Arch
    {
        get => _arch;
    }

    protected string Ext
    {
        get => _ext;
    }

    public CSourceCompiler()
    {
        PlatformSpecific();
        ArchSpecific();
    }

    /// <summary>
    /// compile the source txt, write to the out_path
    /// </summary>
    /// <param name="sourcePath"> c source code</param>
    /// <param name="outPath"> out .so path </param>
    /// <returns> outPath </returns>
    public string Compile(string sourcePath, string outPath)
    {
        var errMsg = new StringBuilder();
        using (var errWriter = new StringWriter(errMsg))
        {
            using (var proc = new Process())
            {
                proc.StartInfo.FileName = Exe;
                proc.StartInfo.Arguments = ArgumentsSpecific(sourcePath, outPath);
                proc.StartInfo.RedirectStandardError = true;
                proc.ErrorDataReceived += (sender, e) => errWriter.WriteLine(e.Data);
                proc.Start();
                proc.BeginErrorReadLine();
                proc.WaitForExit();
                if (proc.ExitCode != 0)
                {
                    throw new InvalidOperationException(errMsg.ToString());
                }
            }
        }
        return outPath;
    }

    /// <summary>
    /// create the temp dll file and compile source
    /// <see cref="Compile(string, string)"/>
    /// </summary>
    public string Compile(string sourcePath) => Compile(sourcePath, CodeGenUtil.GetTempFileName(Ext));
}
#endif
