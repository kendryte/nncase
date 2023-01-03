// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.Diagnostics;

/// <summary>
/// Data dumpper.
/// </summary>
public interface IDumpper
{
    /// <summary>
    /// Gets a value indicating whether dump is enabled.
    /// </summary>
    /// <param name="dumpFlags">Dump flags.</param>
    /// <returns>Whether dump is enabled.</returns>
    bool IsEnabled(DumpFlags dumpFlags);

    /// <summary>
    /// Create sub dummper.
    /// </summary>
    /// <param name="subDirectory">Sub directory.</param>
    /// <returns>Sub dummper.</returns>
    IDumpper CreateSubDummper(string subDirectory);

    void DumpIR(Expr expr, string prefix, string? reletivePath = null);

    FileStream OpenWrite(string reletivePath);
}

public interface IDumpperFactory
{
    IDumpper CreateDummper(string relativePath);
}

/// <summary>
/// Data dump manager.
/// </summary>
public interface IDumpManager
{
    /// <summary>
    /// Gets a value indicating whether dump is enabled.
    /// </summary>
    /// <param name="dumpFlags">Dump flags.</param>
    /// <returns>Whether dump is enabled.</returns>
    bool IsEnabled(DumpFlags dumpFlags);
}

internal sealed class DumpManager : IDumpManager
{
    private readonly CompileOptions _compileOptions;

    public DumpManager(CompileOptions compileOptions)
    {
        _compileOptions = compileOptions;
    }

    //public static bool Append;

    //public static int Count = 1;

    //public static string Dir;

    //public static bool OpenDump { get; private set; }

    //public string CountStr => Count.ToString();

    //public static void RunWithDump(string dir, Action f)
    //{
    //    RunWithDump<int>(dir, () =>
    //    {
    //        f();

    //        // discard return value
    //        return -1;
    //    });
    //}

    //public static T RunWithDump<T>(string dir, Func<T> f)
    //{
    //    Dir = dir;
    //    Count = 1;
    //    OpenDump = true;
    //    Append = false;
    //    var result = f();
    //    OpenDump = false;
    //    return result;
    //}

    //public string GetMaybeDumpDir()
    //{
    //    var root = Path.Join(CompilerServices.CompileOptions.DumpDir, dir);
    //    if (!Directory.Exists(root))
    //    {
    //        Directory.CreateDirectory(root);
    //    }

    //    return root;
    //}

    //protected void UpdateOrder(string root, string target, Shape shape)
    //{
    //    using (var order = new StreamWriter(Path.Join(root, "!out_shape_list"), Append))
    //    {
    //        order.WriteLine($"{target}: {DumpUtility.SerializeShape(shape)}");
    //    }
    //}

    //protected void DumpCallParam(string target, ParameterInfo info, Action<StreamWriter> f)
    //{
    //    var path = Path.Join(GetMaybeDumpDir(), $"{CountStr}${target}${info.Name}");
    //    using (var sr = new StreamWriter(path))
    //    {
    //        f(sr);
    //    }
    //}

    //protected void DumpCall(string target, Shape shape, Action<StreamWriter> f)
    //{
    //    var path = Path.Join(GetMaybeDumpDir(), $"{CountStr}${target}");
    //    using (var sr = new StreamWriter(path))
    //    {
    //        f(sr);
    //    }

    //    UpdateOrder(GetMaybeDumpDir(), target, shape);
    //    Append = true;
    //    ++Count;
    //}

    /// <inheritdoc/>
    public bool IsEnabled(DumpFlags dumpFlags)
    {
        return _compileOptions.DumpFlags.HasFlag(dumpFlags);
    }
}
