// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold call of constants.
/// </summary>
[RuleGenerator]
public partial class FoldConstCall : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsCall(
        "call",
        IsOp<Op>(op => op.CanFoldConstCall),
        IsVArgsRepeat("constArgs", () => IsAlt(IsConst(), IsConstTuple(), IsFixedDimension(), IsFixedShape(), IsFixedPadding(), IsFixedPaddings()))) with
    {
        TypePattern = IsType(x => !(x is InvalidType)),
    };

    private Const GetReplace(Call call, IReadOnlyList<BaseExpr> constArgs)
    {
        // note for egraphs.
        var new_call = call.With(arguments: constArgs.ToArray());
        return (Const)Const.FromValue(new_call.Evaluate()).InheritMetaData(call);
    }
}

/// <summary>
/// Fold shape of.
/// </summary>
[RuleGenerator]
public partial class FoldShapeOf : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsShapeOf(IsWildcard("wc") with { TypePattern = HasFixedShape() });

    private Const GetReplace(Expr wc)
    {
        return Const.FromTensor(wc.CheckedShape.ToValueArray());
    }
}

/// <summary>
/// Fold const <see cref="Dimension"/>.
/// </summary>
[RuleGenerator]
public partial class FoldConstDimension : RewriteRule<DimensionPattern>
{
    /// <inheritdoc/>
    public override DimensionPattern Pattern { get; } = IsDimension(
        name: "dim",
        cond: x => x is not DimConst && x.Operands.Length > 0 && x.Operands.AsValueEnumerable().All(x => x is TensorConst or DimConst)) with
    {
        TypePattern = IsType(x => !(x is InvalidType)),
    };

    private Dimension GetReplace(Dimension dim)
    {
        var value = dim.Evaluate().AsTensor().ToScalar<long>();
        return value;
    }
}
