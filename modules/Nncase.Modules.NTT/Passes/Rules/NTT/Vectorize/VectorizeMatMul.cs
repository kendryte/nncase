// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NTT;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class VectorizedMatMulDevectorizePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsVectorizedMatMul(
            "matMul",
            "caller",
            _ => true,
            PatternMatch.F.Tensors.IsUnpack(
                "devectorize",
                "callee",
                _ => true,
                IsWildcard("lhs")),
            IsTensorConst("rhs"));

    private Expr? GetReplace(Unpack devectorize, VectorizedMatMul matMul, Call caller, Call callee, Expr lhs, Tensor rhs)
    {
        var lhsShape = lhs.CheckedShape;
        var dimInfo = matMul.GetDimInfo(lhsShape.Rank, rhs.Rank);
        (var lhsVectorizeKind, var rhsVectorizeKind) = matMul.GetVectorizeKind(lhsShape.Rank, rhs.Rank);
        if (lhsVectorizeKind == VectorizedMatMul.VectorizeKind.None && rhsVectorizeKind == VectorizedMatMul.VectorizeKind.N && devectorize.Axes == [dimInfo.Lk])
        {
            // If the devectorize is on K, we can bitcast the lhs to element type.
            var newDType = ((VectorType)lhs.CheckedTensorType.DType).ElemType;
            return caller.WithArguments([(VectorizedMatMul.Lhs, IR.F.Tensors.Bitcast(lhs, newDType))]);
        }

        return null;
    }
}
