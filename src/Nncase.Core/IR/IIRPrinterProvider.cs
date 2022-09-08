﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// the symbol for ir printer
/// </summary>
public interface IPrintSymbol
{
    /// <summary>
    /// the full span for this symbol
    /// </summary>
    public StringBuilder Span { get; }

    /// <summary>
    /// the symbol name
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// if this symbol is ref, implict we can use this symbol name.
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
    /// <param name="ILmode">if is print is il or script.</param>
    /// <returns>IrPrinter result.</returns>
    string PrintOp(Op op, IIRPrinterContext context, bool ILmode);

    /// <summary>
    /// if expr is callable will write to {dumpPath}/{prefix}_{callable.name}.{ext}`
    /// else write to {dumpPath}/{prefix}_{expr.Type.name}.il`
    /// </summary>
    /// <param name="expr"></param>
    /// <param name="prefix"></param>
    /// <param name="dumpPath"></param>
    void DumpIR(Expr expr, string prefix, string dumpPath);

    /// <summary>
    /// print ir type.
    /// </summary>
    /// <param name="type"></param>
    /// <returns></returns>
    string Print(IRType type);

    /// <summary>
    /// print ir type.
    /// </summary>
    /// <param name="expr"> the expression</param>
    /// <returns>the string.</returns>
    string Print(Expr expr, bool useScript);
}
