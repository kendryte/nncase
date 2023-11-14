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
    /// Gets dump directory.
    /// </summary>
    string Directory { get; }

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
    /// <param name="dumpFlags">Sub dumpFlags.</param>
    /// <returns>Sub dummper.</returns>
    IDumpper CreateSubDummper(string subDirectory, DumpFlags? dumpFlags = null);

    void DumpIR(Expr expr, string prefix, string? reletivePath = null, bool displayCallable = true);

    void DumpDotIR(Expr expr, string prefix, string? reletivePath = null);

    void DumpCSharpIR(Expr expr, string prefix, string? reletivePath = null);

    void DumpPatternIR(Expr expr, string prefix, string? reletivePath = null);

    void DumpModule(IRModule module, string? reletivePath = null);

    Stream OpenFile(string reletivePath, FileMode fileMode = FileMode.Create);
}

/// <summary>
/// Dumpper factory.
/// </summary>
public interface IDumpperFactory
{
    /// <summary>
    /// Gets root dummper.
    /// </summary>
    IDumpper Root { get; }

    /// <summary>
    /// Creat dumpper.
    /// </summary>
    /// <param name="relativePath">Sub directory.</param>
    /// <returns>Dumpper.</returns>
    IDumpper CreateDummper(string relativePath);
}
