// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CPU.Affine;

[RuleGenerator]
public partial class LowerReduce : RewriteRule<Pattern>
{
    public LowerReduce(string moduleKind = CPUTarget.Kind)
    {
        ModuleKind = moduleKind;
    }

    public string ModuleKind { get; }

    public override Pattern Pattern { get; } = IsCall(
            "call",
            IsOp<IR.CPU.PackedReduce>("op", r => r.ReduceOp != ReduceOp.Mean),
            IsWildcard("input") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") });

    private Expr? GetReplace(Expr call, IR.CPU.PackedReduce op, Expr input)
    {
        var inputShape = input.CheckedShape.ToValueArray();
        var rank = inputShape.Length;
        var domains = IR.F.Affine.Domains(rank);
        var outrank = call.CheckedShape.Rank;
        var results = new AffineRange[outrank];
        {
            var j = 0;
            for (int i = 0; i < rank; i++)
            {
                if (op.Axes.Contains(i))
                {
                    if (op.KeepDims == true)
                    {
                        results[j++] = new AffineRange(0, 1);
                    }
                }
                else
                {
                    results[j++] = new AffineRange(domains[i].Offset, domains[i].Extent);
                }
            }
        }

        var affinemap = new AffineMap(domains, default, results);
        var outBuffer = call.CheckedType switch
        {
            TensorType t => IR.F.Buffer.Uninitialized(t.DType, TIR.MemoryLocation.Data, t.Shape.ToValueArray()),
            DistributedType dt => IR.F.Buffer.Uninitialized(dt.TensorType.DType, TIR.MemoryLocation.Data, dt.TensorType.Shape.ToValueArray(), dt.NdSBP, dt.Placement),
            _ => throw new ArgumentOutOfRangeException(nameof(call)),
        };

        return IR.F.Affine.Grid(ModuleKind)
            .Domain(rank, out var domainVar)
            .Read(input, AffineMap.Identity(rank), out var intile)
            .Write(outBuffer, affinemap, out var outTile)
            .Body(TIR.F.CPU.Reduce(intile, outTile, GetLoadPreviousExpr(op.Axes, domainVar), op.PackedAxes.ToArray(), op.PadedNums.ToArray(), op.Axes, op.KeepDims, op.ReduceOp))
            .Build();
    }

    private Expr GetLoadPreviousExpr(IRArray<int> axes, Expr domainVar)
    {
        Expr? outExpr = null;
        foreach (var axis in axes)
        {
            if (outExpr is null)
            {
                outExpr = IR.F.Math.NotEqual(domainVar[axis][0], 0L);
            }
            else
            {
                outExpr = IR.F.Math.LogicalAnd(outExpr, IR.F.Math.NotEqual(domainVar[axis][0], 0L));
            }
        }

        if (outExpr is null)
        {
            throw new NotSupportedException("reduce axes is empty");
        }

        return outExpr;
    }
}
