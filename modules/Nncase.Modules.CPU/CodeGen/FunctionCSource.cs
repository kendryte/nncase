// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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
/// the csource code compiler.
/// </summary>
public class CSourceCompiler
{
    /// <summary>
    /// compiler exe name.
    /// </summary>
    private string _exe = string.Empty;

    /// <summary>
    /// compiler exe name.
    /// </summary>
    private string _arch = string.Empty;

    /// <summary>
    /// compiler exe name.
    /// </summary>
    private string _ext = string.Empty;

    public CSourceCompiler()
    {
        PlatformSpecific();
        ArchSpecific();
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

    /// <summary>
    /// compile the source txt, write to the out_path.
    /// </summary>
    /// <param name="sourcePath"> c source code.</param>
    /// <param name="outPath"> out .so path. </param>
    /// <returns> outPath. </returns>
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
    /// <see cref="Compile(string, string)"/>.
    /// </summary>
    public string Compile(string sourcePath) => Compile(sourcePath, CodeGenUtil.GetTempFileName(Ext));

    /// <summary>
    /// select current pattern's exe.
    /// </summary>
    /// <exception cref="NotSupportedException"></exception>
    private void PlatformSpecific()
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

    private void ArchSpecific()
    {
        _arch = RuntimeInformation.OSArchitecture switch
        {
            Architecture.X64 => RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "x86-64" : "x86_64",
            Architecture.Arm64 => "arm64",
            _ => throw new NotSupportedException(RuntimeInformation.OSArchitecture.ToString()),
        };
    }

    private string ArgumentsSpecific(string sourcePath, string outPath)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return $"{sourcePath} -nostdlib -static -no-pie -fPIC -march={Arch} -o {outPath}";
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
}

internal sealed class FunctionCSource
{
    public FunctionCSource(string declaration, string implementation)
    {
        Declaration = declaration;
        Implementation = implementation;
    }

    public string Declaration { get; }

    public string Implementation { get; }
}
