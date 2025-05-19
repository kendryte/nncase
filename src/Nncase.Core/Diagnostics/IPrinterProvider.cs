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
/// the symbol for ir printer.
/// </summary>
public interface IPrintSymbol
{
    /// <summary>
    /// Gets the full span for this symbol.
    /// </summary>
    string Span { get; }

    /// <summary>
    /// Gets the symbol name.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets a value indicating whether if this symbol is ref, implict we can use this symbol name.
    /// </summary>
    bool IsRefSymobl { get; }
}

/// <summary>
/// Type inference provider interface.
/// </summary>
public interface IPrinterProvider
{
    /// <summary>
    /// IrPrinter operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">IrPrinter context.</param>
    /// <returns>IrPrinter result.</returns>
    string? PrintOp(Op op, IPrintOpContext context);

    /// <summary>
    /// if expr is callable will write to {dumpPath}/{prefix}_{callable.name}.{ext}`
    /// else write to {dumpPath}/{prefix}_{expr.Type.name}.il`.
    /// </summary>
    void DumpIR(BaseExpr expr, string prefix, string dumpPath, PrinterFlags flags);

    /// <summary>
    /// print ir type.
    /// </summary>
    string Print(IRType type, PrinterFlags flags);

    /// <summary>
    /// print ir type.
    /// </summary>
    /// <param name="expr"> the expression.</param>
    /// <param name="flags"> display callable.</param>
    /// <returns>the string.</returns>
    string Print(BaseExpr expr, PrinterFlags flags);

    void DumpDotIR(BaseExpr expr, string prefix, string dumpPath, PrinterFlags flags);

    void DumpCSharpIR(BaseExpr expr, string prefix, string dumpDir, bool randConst);

    void DumpPatternIR(BaseExpr expr, string prefix, string dumpDir);
}
