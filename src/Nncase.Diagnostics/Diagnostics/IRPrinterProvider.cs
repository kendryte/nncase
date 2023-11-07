// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Diagnostics;

/// <summary>
/// IR printer.
/// </summary>
internal sealed class IRPrinterProvider : IIRPrinterProvider
{
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the <see cref="IRPrinterProvider"/> class.
    /// ctor.
    /// </summary>
    /// <param name="serviceProvider"> compiler servicer provider.</param>
    public IRPrinterProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    /// <inheritdoc/>
    public void DumpIR(Expr expr, string prefix, string dumpPath, bool display_callable)
    {
        var nprefix = prefix.Any() ? prefix + "_" : prefix;
        string ext = expr is PrimFunction ? "script" : "il";
        string name = expr is Callable c ? c.Name : expr.GetType().Name;
        string file_path = Path.Combine(dumpPath, $"{nprefix}{name}.{ext}");
        if (string.IsNullOrEmpty(dumpPath))
        {
            throw new ArgumentException("The dumpPath Is Empty!", nameof(dumpPath));
        }

        Directory.CreateDirectory(dumpPath);

        using var dumpFile = File.Open(file_path, FileMode.Create);
        using var dumpWriter = new StreamWriter(dumpFile);
        switch (expr)
        {
            case PrimFunction pf:
                new ScriptPrintVisitor(dumpWriter, display_callable).Visit(pf);
                break;
            default:
                new ILPrintVisitor(dumpWriter, display_callable, 0).Visit(expr);
                break;
        }
    }

    /// <inheritdoc/>
    public void DumpDotIR(Expr expr, string prefix, string dumpDir, bool display_callable)
    {
        if (string.IsNullOrEmpty(dumpDir))
        {
            throw new ArgumentException("The dumpPath Is Empty!", nameof(dumpDir));
        }

        Directory.CreateDirectory(dumpDir);

        string name = expr is Callable c ? c.Name : expr.GetType().Name;

        var visitor = new ILDotPrintVisitor(display_callable);
        visitor.Visit(expr);
        visitor.SaveToFile(name, prefix, dumpDir);
    }

    /// <inheritdoc/>
    public void DumpCSharpIR(Expr expr, string prefix, string dumpDir, bool randConst)
    {
        var nprefix = prefix.Any() ? prefix + "_" : prefix;
        string ext = "cs";
        string name = expr is Callable c ? c.Name : expr.GetType().Name;
        string file_path = Path.Combine(dumpDir, $"{nprefix}{name}.{ext}");
        string rdata_path = Path.Combine(dumpDir, $"{nprefix}{name}.bin");
        if (string.IsNullOrEmpty(dumpDir))
        {
            throw new ArgumentException("The dumpDir Is Empty!");
        }

        Directory.CreateDirectory(dumpDir);

        using var dumpFile = File.Open(file_path, FileMode.Create);
        using var dumpWriter = new StreamWriter(dumpFile);
        if (!randConst)
        {
            using var rdataFile = File.Open(rdata_path, FileMode.Create);
            using var rdataWriter = new BinaryWriter(rdataFile);
            new CSharpPrintVisitor(dumpWriter, rdataWriter, 0, randConst, true).Visit(expr);
        }
        else
        {
            new CSharpPrintVisitor(dumpWriter, BinaryWriter.Null, 0, randConst, true).Visit(expr);
        }
    }

    /// <inheritdoc/>
    public void DumpPatternIR(Expr expr, string prefix, string dumpDir)
    {
        var nprefix = prefix.Any() ? prefix + "_" : prefix;
        string ext = "cs";
        string name = expr is Callable c ? c.Name : expr.GetType().Name;
        string file_path = Path.Combine(dumpDir, $"{nprefix}{name}.{ext}");
        if (string.IsNullOrEmpty(dumpDir))
        {
            throw new ArgumentException("The dumpDir Is Empty!");
        }

        Directory.CreateDirectory(dumpDir);

        using var dumpFile = File.Open(file_path, FileMode.Create);
        using var dumpWriter = new StreamWriter(dumpFile);
        new PatternPrintVisitor(dumpWriter, 0).Visit(expr);
    }

    /// <inheritdoc/>
    public string Print(IRType type)
    {
        var sb = new StringBuilder();
        using var dumpWriter = new StringWriter(sb);
        return new ILPrintVisitor(dumpWriter, true, 0).VisitType(type);
    }

    /// <inheritdoc/>
    public string Print(Expr expr, bool useScript)
    {
        var sb = new StringBuilder();
        using var dumpWriter = new StringWriter(sb);
        var text = expr is PrimFunction || useScript
            ? new ScriptPrintVisitor(dumpWriter, true).Visit(expr).Serialize()
            : new ILPrintVisitor(dumpWriter, true, 0).Visit(expr);

        return useScript ? text : expr switch
        {
            Const or None or Var or Op => text,
            _ => sb.ToString(),
        };
    }

    /// <inheritdoc/>
    public string PrintOp(Op op, IIRPrinterContext context, bool iLmode)
    {
        // TODO: Add printers cache.
        var irprinterType = typeof(IOpPrinter<>).MakeGenericType(op.GetType());
        if (_serviceProvider.GetService(irprinterType) is IOpPrinter irprinter)
        {
            return irprinter.Visit(context, op, iLmode);
        }

        return $"{context.Get(op)}({string.Join(", ", context.GetArguments(op).Select(s => s.ToString()))})";
    }
}
