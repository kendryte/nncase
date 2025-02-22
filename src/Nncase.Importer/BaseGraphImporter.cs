// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase;

public record class GraphImportResult(Var[] Inputs, Expr Outputs);

/// <summary>
/// Importer base.
/// </summary>
public abstract class BaseGraphImporter
{
    private readonly SortedSet<string> _opsInModel = new();
    private readonly SortedSet<string> _unsupportedOp = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="BaseGraphImporter"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    /// <param name="compileSession">Compile session.</param>
    /// <param name="module">IRModule.</param>
    public BaseGraphImporter(string name, CompileSession compileSession, IRModule module)
    {
        Name = name;
        CompileSession = compileSession;
        IRModule = module;
        Dumpper = DumpScope.GetCurrent(compileSession).CreateSubDummper(name, null);
    }

    public string Name { get; }

    /// <summary>
    /// Gets compile session.
    /// </summary>
    protected CompileSession CompileSession { get; }

    /// <summary>
    /// Gets IRModule.
    /// </summary>
    protected IRModule IRModule { get; }

    /// <summary>
    /// Gets dumpper.
    /// </summary>
    protected IDumpper Dumpper { get; }

    protected List<Var> Inputs { get; set; } = new();

    /// <summary>
    /// import the model.
    /// </summary>
    /// <returns>IRModule.</returns>
    public GraphImportResult Import()
    {
        ConvertOp();
        SupportedCheck(GetType().Name.Split("Importer")[0]);
        var outputs = CreateOutputs();

        if (Dumpper.IsEnabled(DumpFlags.ImportOps))
        {
            DumpOpsInModel(Dumpper.OpenFile("OpsInModel.txt"));
        }

        var result = new GraphImportResult(Inputs.ToArray(), outputs);

        // GC here as large models often leave much garbage.
        GC.Collect();
        return result;
    }

    protected void AddToOutputs<TKey, TNode>(Dictionary<TKey, Expr> outTensors, TKey[] opOutputs, TNode output)
        where TKey : notnull
    {
        var outLength = opOutputs.Length;
        if (output is Expr expr)
        {
            if (opOutputs.Length == 1)
            {
                outTensors.Add(opOutputs[0], expr);
            }
            else
            {
                for (int i = 0; i < outLength; i++)
                {
                    outTensors.Add(opOutputs[i], IR.F.Tensors.GetItem(expr, i));
                }
            }
        }
        else if (output is IReadOnlyList<Expr> exprs)
        {
            Trace.Assert(outLength == exprs.Count, $"Op outputs length should be {outLength}.");
            for (int i = 0; i < outLength; i++)
            {
                outTensors.Add(opOutputs[i], exprs[i]);
            }
        }
        else
        {
            throw new InvalidOperationException("Visit result is not expression(s).");
        }
    }

    protected abstract void ConvertOp();

    protected abstract Expr CreateOutputs();

    protected Expr UnSupportedOp(string opType)
    {
        _unsupportedOp.Add(opType);
        return None.Default;
    }

    protected void AddOpInModel(string opType)
    {
        _opsInModel.Add(opType);
    }

    protected void SupportedCheck(string name)
    {
        if (_unsupportedOp.Count > 0)
        {
            throw new NotSupportedException(
                $"Not Supported {name} op: {string.Join(',', _unsupportedOp)}");
        }
    }

    protected Function AddFunction(string name, Var[] inputs, Expr body)
    {
        var func = new Function(name, body, inputs);
        IRModule.Add(func);
        return func;
    }

    private void DumpOpsInModel(Stream path)
    {
        using var sr = new StreamWriter(path);
        foreach (var op in _opsInModel)
        {
            sr.WriteLine(op);
        }
    }
}
