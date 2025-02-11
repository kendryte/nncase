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
internal sealed class IRPrinterProvider : IPrinterProvider
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
    public void DumpIR(Expr expr, string prefix, string dumpPath, PrinterFlags flags)
    {
        var nprefix = prefix.Any() ? prefix + "_" : prefix;
        string ext = flags.HasFlag(PrinterFlags.Script) ? "script" : "il";
        string name = expr is Callable c ? c.Name : expr.GetType().Name;
        string filePath = Path.Combine(dumpPath, $"{nprefix}{name}.{ext}");
        if (string.IsNullOrEmpty(dumpPath))
        {
            throw new ArgumentException("The dumpPath Is Empty!", nameof(dumpPath));
        }

        Directory.CreateDirectory(dumpPath);

        using var dumpFile = File.Open(filePath, FileMode.Create);
        using var dumpWriter = new IndentedWriter(dumpFile);
        switch (ext)
        {
            case "script":
                new ScriptPrintVisitor(dumpWriter, flags).Visit(expr);
                break;
            default:
                new ILPrintVisitor(dumpWriter, flags, new Dictionary<Expr, string>()).Visit(expr);
                break;
        }
    }

    /// <inheritdoc/>
    public void DumpDotIR(Expr expr, string prefix, string dumpDir, PrinterFlags flags)
    {
        if (string.IsNullOrEmpty(dumpDir))
        {
            throw new ArgumentException("The dumpPath Is Empty!", nameof(dumpDir));
        }

        Directory.CreateDirectory(dumpDir);

        string name = expr is Callable c ? c.Name : expr.GetType().Name;

        var visitor = new ILDotPrintVisitor(flags);
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
    public string Print(IRType type, PrinterFlags flags)
    {
        var stream = Stream.Null;
        using var dumpWriter = new IndentedWriter(stream);
        return new ILPrintVisitor(dumpWriter, flags, new Dictionary<Expr, string>()).VisitType(type);
    }

    /// <inheritdoc/>
    public string Print(Expr expr, PrinterFlags flags)
    {
        using var stream = new MemoryStream(128 * 1024 * 1024);

        using (var writer = new IndentedWriter(stream))
        {
            if (flags.HasFlag(PrinterFlags.Script))
            {
                new ScriptPrintVisitor(writer, flags).Visit(expr);
            }
            else
            {
                new ILPrintVisitor(writer, flags, new Dictionary<Expr, string>()).Visit(expr);
            }
        }

        stream.Seek(0, SeekOrigin.Begin);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    /// <inheritdoc/>
    public string? PrintOp(Op op, IPrintOpContext context)
    {
        // TODO: Add printers cache.
        var irprinterType = typeof(IOpPrinter<>).MakeGenericType(op.GetType());
        if (_serviceProvider.GetService(irprinterType) is IOpPrinter irprinter)
        {
            return irprinter.Visit(context, op);
        }

        return null;
    }
}
