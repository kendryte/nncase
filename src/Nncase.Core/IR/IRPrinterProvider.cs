// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR.Math;
using Nncase.TIR;

namespace Nncase.IR;



/// <summary>
/// IR printer.
/// </summary>
public sealed class IRPrinterProvider : IIRPrinterProvider
{

    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// ctor.
    /// </summary>
    /// <param name="serviceProvider"> compiler servicer provider</param>
    public IRPrinterProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    /// <inheritdoc/>
    public void DumpIR(Expr expr, string prefix, string dumpPath)
    {
        var nprefix = prefix.Any() ? prefix + "_" : prefix;
        string ext = expr is Function ? "il" : "script";
        string name = expr is Callable c ? c.Name : expr.GetType().Name;
        string file_path = Path.Combine(dumpPath, $"{nprefix}{name}.{ext}");
        Directory.CreateDirectory(dumpPath);

        using var dumpFile = File.Open(file_path, FileMode.Create);
        using var dumpWriter = new StreamWriter(dumpFile);
        switch (expr)
        {
            case PrimFunction pf:
                new ScriptPrintVisitor(dumpWriter).Visit(pf);
                break;
            default:
                new ILPrintVisitor(dumpWriter).Visit(expr);
                break;
        }
    }

    /// <inheritdoc/>
    public string Print(IRType type)
    {
        var sb = new StringBuilder();
        using var dumpWriter = new StringWriter(sb);
        return new ILPrintVisitor(dumpWriter).VisitType(type);
    }

    /// <inheritdoc/>
    public string Print(Expr expr)
    {
        var sb = new StringBuilder();
        using var dumpWriter = new StringWriter(sb);
        return expr switch
        {
            PrimFunction pf => new ScriptPrintVisitor(dumpWriter).Visit(pf).Serialize(),
            _ => new ILPrintVisitor(dumpWriter).Visit(expr)
        };
    }

    /// <inheritdoc/>
    public string PrintOp(Op op, IIRPrinterContext context, bool ILmode)
    {
        // TODO: Add printers cache.
        var irprinterType = typeof(IOpPrinter<>).MakeGenericType(op.GetType());
        if (_serviceProvider.GetService(irprinterType) is IOpPrinter irprinter)
        {
            return irprinter.Visit(context, op, ILmode);
        }
        return $"{context.Get(op)}({string.Join(", ", context.GetArguments(op).Select(s => s.ToString()))})";
    }

}
