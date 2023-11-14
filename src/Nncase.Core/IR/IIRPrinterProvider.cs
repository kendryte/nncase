// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// the symbol for ir printer.
/// </summary>
public interface IPrintSymbol
{
    /// <summary>
    /// Gets the full span for this symbol.
    /// </summary>
    public StringBuilder Span { get; }

    /// <summary>
    /// Gets the symbol name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets a value indicating whether if this symbol is ref, implict we can use this symbol name.
    /// </summary>
    public bool IsRefSymobl { get; }

    /// <summary>
    /// to string.
    /// </summary>
    /// <returns> string. </returns>
    public string Serialize();
}

/// <summary>
/// Type inference provider interface.
/// </summary>
public interface IIRPrinterProvider
{
    /// <summary>
    /// IrPrinter operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">IrPrinter context.</param>
    /// <param name="iLmode">if is print is il or script.</param>
    /// <returns>IrPrinter result.</returns>
    string PrintOp(Op op, IIRPrinterContext context, bool iLmode);

    /// <summary>
    /// if expr is callable will write to {dumpPath}/{prefix}_{callable.name}.{ext}`
    /// else write to {dumpPath}/{prefix}_{expr.Type.name}.il`.
    /// </summary>
    void DumpIR(Expr expr, string prefix, string dumpPath, bool display_callable);

    /// <summary>
    /// if expr is callable will write to {dumpPath}/{prefix}_{callable.name}.dot`.
    /// <remarks>
    /// not support prim func/prim func wrapper.
    /// </remarks>
    /// </summary>
    void DumpDotIR(Expr expr, string prefix, string dumpPath, bool display_callable);

    /// <summary>
    /// dump the expr as csharp code.
    /// </summary>
    /// <param name="expr">expression.</param>
    /// <param name="prefix">file prefix.</param>
    /// <param name="dumpDir">file dump ir.</param>
    /// <param name="randConst">randConst = false will save the const into bin.</param>
    public void DumpCSharpIR(Expr expr, string prefix, string dumpDir, bool randConst);

    /// <summary>
    /// dump the expr as csharp code.
    /// </summary>
    /// <param name="expr">expression.</param>
    /// <param name="prefix">file prefix.</param>
    /// <param name="dumpDir">file dump ir.</param>
    public void DumpPatternIR(Expr expr, string prefix, string dumpDir);

    /// <summary>
    /// print ir type.
    /// </summary>
    string Print(IRType type);

    /// <summary>
    /// print ir type.
    /// </summary>
    /// <param name="expr"> the expression.</param>
    /// <param name="useScript"> tir mode.</param>
    /// <returns>the string.</returns>
    string Print(Expr expr, bool useScript);
}
