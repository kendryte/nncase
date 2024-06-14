// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.ArgsStruct;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerReduction : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsReduce(
        "reduce",
        "reduceCall",
        _ => true,
        IsWildcard("input") with { TypePattern = IsFloat() & HasFixedShape() },
        IsTensorConst("axis"),
        IsTensorConst("initValue"),
        IsTensorConst("keepDims"));

    public virtual int GetOpT()
    {
        return -1;
    }

    private Expr? GetReplace(Reduce reduce, Expr input, long[] axis, Expr initValue, bool keepDims)
    {
        // TODO: split input
        if ((input.CheckedShape.ToList()[0] != 1 && input.CheckedShape.Rank == 4) || input.CheckedShape.Rank > 4 || input.CheckedDataType == DataTypes.Float16)
        {
            return null;
        }

        // Support other reduction ops combined with several ops.
        var otherOpType = GetOpT();
        var reductionType = otherOpType == -1
            ? reduce.ReduceOp switch
            {
                ReduceOp.Sum => 0,
                ReduceOp.Mean => 3,
                ReduceOp.Max => 4,
                ReduceOp.Min => 5,
                ReduceOp.Prod => 6,
                _ => -1,
            }
            : otherOpType;
        if (reductionType == -1)
        {
            return null;
        }

        var newAxis = axis;

        var newInput = input;
        var newInputVar = new Var(newInput.CheckedType);
        if (input.CheckedShape.Rank == 4)
        {
            if (axis.Length == input.CheckedShape.Rank)
            {
                // 排除batch维度
                newAxis = newAxis.Remove(0);
                newAxis = newAxis.Remove(-input.CheckedShape.Rank);
            }

            for (int i = 0; i < newAxis.Length; i++)
            {
                if (newAxis[i] == 0 || newAxis[i] > 4 || newAxis[i] < -3)
                {
                    return null;
                }

                newAxis[i] = newAxis[i] > 0 ? newAxis[i] - 1 : newAxis[i];
            }

            newInput = Squeeze(input, new[] { 0 });
            newInputVar = new Var(newInput.CheckedType);
        }

        var args = new ReductionArgs(reductionType, newAxis.Length == newInput.CheckedShape.Rank ? 1 : 0, 0, newAxis, keepDims ? 1 : 0);

        var pool = new Call(new Fusion("ncnn", NcnnReduction(newInputVar, args), new[] { newInputVar }), newInput);

        if (input.CheckedShape.Rank == 4)
        {
            return Unsqueeze(pool, new[] { 0 });
        }
        else
        {
            return pool;
        }
    }
}

[RuleGenerator]
public partial class LowerReductionSumSquare : LowerReduction
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } =
            IsReduce(
                "reduce",
                "reduceCall",
                ReduceOp.Sum,
                IsUnary(
                    "square",
                    UnaryOp.Square,
                    IsWildcard("input") with { TypePattern = IsFloat() & HasFixedShape() }),
                IsTensorConst("axis"),
                IsTensorConst("initValue"),
                IsTensorConst("keepDims"));

    public override int GetOpT()
    {
        return 2;
    }
}

[RuleGenerator]
public partial class LowerReductionL1 : LowerReduction
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } =
        IsReduce(
            "reduce",
            "reduceCall",
            ReduceOp.Sum,
            IsUnary(
                "abs",
                UnaryOp.Abs,
                IsWildcard("input") with { TypePattern = IsFloat() & HasFixedShape() }),
            IsTensorConst("axis"),
            IsTensorConst("initValue"),
            IsTensorConst("keepDims"));

    public override int GetOpT()
    {
        return 7;
    }
}

[RuleGenerator]
public class LowerReductionL2 : LowerReduction
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } =
        IsUnary(
            "sqrt",
            UnaryOp.Sqrt,
            IsReduce(
                "reduce",
                "reduceCall",
                ReduceOp.Sum,
                IsUnary(
                    "square",
                    UnaryOp.Square,
                    IsWildcard("input") with { TypePattern = IsFloat() & HasFixedShape() }),
                IsTensorConst("axis"),
                IsTensorConst("initValue"),
                IsTensorConst("keepDims")));

    public override int GetOpT()
    {
        return 8;
    }
}

[RuleGenerator]
public partial class LowerReductionLogSum : LowerReduction
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } =
        IsUnary(
            "log",
            UnaryOp.Log,
            IsReduce(
                "reduce",
                "reduceCall",
                ReduceOp.Sum,
                IsWildcard("input") with { TypePattern = IsFloat() & HasFixedShape() },
                IsTensorConst("axis"),
                IsTensorConst("initValue"),
                IsTensorConst("keepDims")));

    public override int GetOpT()
    {
        return 9;
    }
}

[RuleGenerator]
public partial class LowerReductionLogSumExp : LowerReduction
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } =
        IsUnary(
            "log",
            UnaryOp.Log,
            IsReduce(
                "reduce",
                "reduceCall",
                ReduceOp.Sum,
                IsUnary(
                    "exp",
                    UnaryOp.Exp,
                    IsWildcard("input") with { TypePattern = IsFloat() & HasFixedShape() }),
                IsTensorConst("axis"),
                IsTensorConst("initValue"),
                IsTensorConst("keepDims")));

    public override int GetOpT()
    {
        return 10;
    }
}
