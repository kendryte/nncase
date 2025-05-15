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
using Nncase.IR.Shapes;

namespace Nncase;

/// <summary>
/// Importer base.
/// </summary>
public abstract class BaseImporter
{
    private readonly SortedSet<string> _opsInModel = new();
    private readonly SortedSet<string> _unsupportedOp = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="BaseImporter"/> class.
    /// </summary>
    /// <param name="compileSession">Compile session.</param>
    public BaseImporter(CompileSession compileSession)
    {
        CompileSession = compileSession;
        Dumpper = DumpScope.GetCurrent(compileSession).CreateSubDummper("Import", null);
    }

    /// <summary>
    /// Gets compile session.
    /// </summary>
    protected CompileSession CompileSession { get; }

    /// <summary>
    /// Gets dumpper.
    /// </summary>
    protected IDumpper Dumpper { get; }

    /// <summary>
    /// import the model.
    /// </summary>
    /// <returns>IRModule.</returns>
    public IRModule Import()
    {
        var (inputs, varMap) = CreateInputs();
        ConvertOp();
        SupportedCheck(GetType().Name.Split("Importer")[0]);
        var outputs = CreateOutputs();

        if (Dumpper.IsEnabled(DumpFlags.ImportOps))
        {
            DumpOpsInModel(Dumpper.OpenFile("OpsInModel.txt"));
        }

        var module = CreateModule(inputs.ToArray(), varMap, outputs);

        // GC here as large models often leave much garbage.
        GC.Collect();
        return module;
    }

    protected void AddToOutputs<TKey, TNode>(Dictionary<TKey, BaseExpr> outTensors, TKey[] opOutputs, TNode output)
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
        else if (output is IReadOnlyList<BaseExpr> exprs)
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

    protected abstract (IEnumerable<Var> Inputs, Dictionary<Var, Dimension[]> VarMap) CreateInputs();

    protected abstract void ConvertOp();

    protected abstract BaseExpr CreateOutputs();

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

    protected T GetInputExpr<T>(BaseExpr expr)
        where T : BaseExpr
    {
        var type = typeof(T);
        if (type == typeof(BaseExpr))
        {
            return (T)expr;
        }
        else if (type == typeof(Expr))
        {
            return (T)expr;
        }
        else if (type == typeof(Dimension))
        {
            return (T)(BaseExpr)expr.AsDim();
        }
        else if (type == typeof(Shape) || type == typeof(RankedShape))
        {
            return (T)(BaseExpr)expr.AsShape();
        }
        else if (type == typeof(Padding))
        {
            return (T)(BaseExpr)expr.AsPadding();
        }
        else if (type == typeof(Paddings))
        {
            return (T)(BaseExpr)GetPaddings((Expr)expr);
        }
        else
        {
            throw new InvalidOperationException($"Unsupported type: {type}.");
        }
    }

    /// <summary>
    /// Get paddings from expr.
    /// The paddings is a 2D tensor, shape = [channels, 2(before, after)].
    /// </summary>
    /// <param name="expr">Expr.</param>
    protected Paddings GetPaddings(Expr expr)
    {
        var shape = expr.CheckedShape;
        if (!shape.IsFixed || shape.Rank != 2)
        {
            throw new InvalidDataException($"Paddings should be a 2D tensor, but got {shape}.");
        }

        var channels = (int)shape[0].FixedValue;
        return new Paddings(Enumerable
            .Range(0, channels)
            .Select(i =>
            {
                var before = expr[i, 0].AsDim();
                var after = expr[i, 1].AsDim();
                return new Padding(before, after);
            })
            .ToArray());
    }

    private IRModule CreateModule(Var[] inputs, Dictionary<Var, Dimension[]> varMap, BaseExpr body)
    {
        var mainFunc = new Function("main", body, inputs, varMap);
        var module = new IRModule(mainFunc);
        return module;
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
