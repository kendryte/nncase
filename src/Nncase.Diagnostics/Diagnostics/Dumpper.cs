// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Diagnostics;

internal sealed class Dumpper : IDumpper
{
    private readonly DumpFlags _dumpFlags;
    private readonly string _dumpDirectory;

    public Dumpper(DumpFlags dumpFlags, string dumpDirectory)
    {
        _dumpFlags = dumpFlags;
        _dumpDirectory = dumpDirectory;
    }

    public string Directory => _dumpDirectory;

    public IDumpper CreateSubDummper(string subDirectory, DumpFlags? dumpFlags)
    {
        DumpFlags subDumpFlags = _dumpFlags;
        if (dumpFlags is DumpFlags flags)
        {
            subDumpFlags &= flags; // the sub dumpFlags must less than parents.
        }

        return new Dumpper(subDumpFlags, Path.Join(_dumpDirectory, subDirectory));
    }

    public void DumpIR(Expr expr, string prefix, string? reletivePath = null, bool displayCallable = true)
    {
        var path = Path.Join(_dumpDirectory, reletivePath);
        CompilerServices.DumpIR(expr, prefix, EnsureWritable(path), displayCallable);
    }

    public void DumpDotIR(Expr expr, string prefix, string? reletivePath = null)
    {
        var path = Path.Join(_dumpDirectory, reletivePath);
        CompilerServices.DumpDotIR(expr, prefix, EnsureWritable(path));
    }

    public void DumpCSharpIR(Expr expr, string prefix, string? reletivePath = null)
    {
        var path = Path.Join(_dumpDirectory, reletivePath);
        CompilerServices.DumpCSharpIR(expr, prefix, EnsureWritable(path));
    }

    public void DumpModule(IRModule module, string? reletivePath = null)
    {
        foreach (var func in module.Functions)
        {
            DumpIR(func, string.Empty, reletivePath, false);
        }
    }

    public bool IsEnabled(DumpFlags dumpFlags)
    {
        return _dumpFlags.HasFlag(dumpFlags);
    }

    public Stream OpenFile(string reletivePath, FileMode fileMode)
    {
        var path = Path.Join(_dumpDirectory, reletivePath);
        return File.Open(EnsureWritable(path), fileMode);
    }

    public override string ToString() => $"Dumpper({_dumpFlags})";

    private static string EnsureWritable(string path)
    {
        var directory = Path.GetDirectoryName(path) ?? throw new ArgumentException($"Invalid path: {path}");
        System.IO.Directory.CreateDirectory(directory);
        return path;
    }
}

internal sealed class DumpperFactory : IDumpperFactory
{
    private readonly CompileOptions _compileOptions;

    public DumpperFactory(CompileOptions compileOptions)
    {
        _compileOptions = compileOptions;
    }

    public IDumpper Root => new Dumpper(_compileOptions.DumpFlags, _compileOptions.DumpDir);

    public IDumpper CreateDummper(string relativePath)
    {
        return new Dumpper(_compileOptions.DumpFlags, Path.Join(_compileOptions.DumpDir, relativePath));
    }
}
