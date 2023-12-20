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

namespace Nncase.CodeGen.CPU;

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
                proc.StartInfo.WorkingDirectory = Directory.GetCurrentDirectory();
                proc.StartInfo.RedirectStandardError = true;
                proc.StartInfo.RedirectStandardOutput = true;
                proc.OutputDataReceived += (sender, e) => errWriter.WriteLine(e.Data);
                proc.ErrorDataReceived += (sender, e) => errWriter.WriteLine(e.Data);
                proc.Start();
                proc.BeginErrorReadLine();
                proc.BeginOutputReadLine();
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
    public string Compile(string sourcePath) => Compile(sourcePath, Path.Join(sourcePath, "build", Path.GetFileName(sourcePath)));

    private static string? FindVCVarPath()
    {
        var vsDir = Environment.GetEnvironmentVariable("VSAPPIDDIR");
        if (!string.IsNullOrEmpty(vsDir))
        {
            return Path.Combine(vsDir, "..\\..\\VC\\Auxiliary\\Build\\vcvarsall.bat");
        }
        else
        {
            var vsWhereDir = Path.Combine(Environment.GetEnvironmentVariable("ProgramFiles(x86)")!, "Microsoft Visual Studio\\Installer\\vswhere");
            if (string.IsNullOrEmpty(vsWhereDir))
            {
                return null;
            }

            using (var proc = new Process())
            {
                proc.StartInfo.FileName = vsWhereDir;
                proc.StartInfo.Arguments = "-prerelease -latest -property installationPath";
                proc.StartInfo.RedirectStandardOutput = true;
                proc.Start();
                proc.BeginOutputReadLine();
                proc.WaitForExit();
                vsDir = proc.StandardOutput.ReadToEnd();
                return Path.Combine(vsDir, "VC\\Auxiliary\\Build\\vcvarsall.bat");
            }
        }
    }

    /// <summary>
    /// select current pattern's exe.
    /// </summary>
    /// <exception cref="NotSupportedException">NotSupportedException.</exception>
    private void PlatformSpecific()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            _exe = "/bin/bash";
            _ext = "so";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            _exe = "/bin/bash";
            _ext = "dylib";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            _exe = "cmd";
            _ext = "dll";
        }

        if (System.Environment.GetEnvironmentVariable("NNCASE_CPU_COMPILER") is string exe)
        {
            _exe = exe;
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
        var config = "RelWithDebInfo";
        var script = $"""
            cd {sourcePath} &&
            cmake -E remove_directory build &&
            cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE={config} &&
            cmake --build build --config {config}
            """.Replace("\r\n", " ", StringComparison.Ordinal);

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return $"-c \"{script}\"";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return $"-c \"{script}\"";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var vcVarPath = FindVCVarPath();
            if (!string.IsNullOrEmpty(vcVarPath))
            {
                return $"/C \"(\"{vcVarPath}\" x64) && {script}\"";
            }

            return $"/C {script}";
        }

        throw new NotSupportedException("Only Support Linux/Osx/Windows");
    }
}
