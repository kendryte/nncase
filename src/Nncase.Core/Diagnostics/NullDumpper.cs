// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Diagnostics;

/// <summary>
/// <see cref="IDumpper"/> with no backing store.
/// </summary>
public sealed class NullDumpper : IDumpper
{
    /// <summary>
    /// Gets instance.
    /// </summary>
    public static NullDumpper Instance { get; } = new NullDumpper();

    /// <inheritdoc/>
    public string Directory => System.IO.Directory.GetCurrentDirectory();

    /// <inheritdoc/>
    public IDumpper CreateSubDummper(string subDirectory, DumpFlags? dumpFlags) => this;

    /// <inheritdoc/>
    public void DumpIR(Expr expr, string prefix, string? reletivePath = null, bool displayCallable = true)
    {
    }

    /// <inheritdoc/>
    public void DumpDotIR(Expr expr, string prefix, string? reletivePath = null)
    {
    }

    /// <inheritdoc/>
    public void DumpModule(IRModule module, string? reletivePath = null)
    {
    }

    /// <inheritdoc/>
    public void DumpCSharpIR(Expr expr, string prefix, string? reletivePath = null)
    {
    }

    /// <inheritdoc/>
    public void DumpPatternIR(Expr expr, string prefix, string? reletivePath = null)
    {
    }

    /// <inheritdoc/>
    public bool IsEnabled(DumpFlags dumpFlags) => false;

    /// <inheritdoc/>
    public Stream OpenFile(string reletivePath, FileMode fileMode = FileMode.Create) => Stream.Null;
}
