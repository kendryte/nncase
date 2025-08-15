// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Decompose QLinearMatMul.
/// </summary>
[RuleGenerator]
public sealed partial class DecomposeQLinearMatMul : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsQLinearMatMul(
            "qLinearMatMul",
            "qLinearMatMulCall",
            _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhs"),
            IsWildcard("lhsScale"),
            IsWildcard("lhsZeroPoint"),
            IsWildcard("rhsScale"),
            IsWildcard("rhsZeroPoint"),
            IsWildcard("outputScale"),
            IsWildcard("outputZeroPoint"));

    private Expr? GetReplace(Op qLinearMatMul, Call qLinearMatMulCall, Expr lhs, Expr rhs, Expr lhsScale, Expr lhsZeroPoint, Expr rhsScale, Expr rhsZeroPoint, Expr outputScale, Expr outputZeroPoint)
    {
        if (lhs.CheckedDataType == DataTypes.Int8)
        {
            // assume zero point is zero
            var matmul = IR.F.Math.MatMul(lhs, rhs, outputScale.CheckedDataType);
            var output = (matmul * lhsScale * rhsScale * outputScale) + IR.F.Tensors.Cast(outputZeroPoint, outputScale.CheckedDataType);
            return output;
        }
        else
        {
            var lhsDeq = IR.F.Tensors.Cast(lhs - IR.F.Tensors.Cast(lhsZeroPoint, lhs.CheckedDataType), lhsScale.CheckedDataType) * lhsScale;
            var rhsDeq = IR.F.Tensors.Cast(rhs - IR.F.Tensors.Cast(rhsZeroPoint, rhs.CheckedDataType), rhsScale.CheckedDataType) * rhsScale;
            var matmul = IR.F.Math.MatMul(lhsDeq, rhsDeq, outputScale.CheckedDataType);
            var output = (matmul * outputScale) + IR.F.Tensors.Cast(outputZeroPoint, outputScale.CheckedDataType);
            return output;
        }
    }
}
