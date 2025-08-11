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
public sealed partial class PackMatMulByN : RewriteRule<Pattern>
{
    private readonly int _nr;

    public PackMatMulByN(int nr)
    {
        _nr = nr;
    }

    public override Pattern Pattern { get; } =
        IsVectorizedMatMul(
            "matMul",
            "caller",
            _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhs"));

    private Expr? GetReplace(VectorizedMatMul matMul, Call caller, Expr lhs, Expr rhs)
    {
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var dimInfo = matMul.GetDimInfo(lhsShape.Rank, rhsShape.Rank);
        (var lhsVectorizeKind, var rhsVectorizeKind) = matMul.GetVectorizeKind(lhsShape.Rank, rhsShape.Rank);
        if (lhsVectorizeKind == VectorizedMatMul.VectorizeKind.None && rhsVectorizeKind == VectorizedMatMul.VectorizeKind.N
            && !matMul.TransposeA && !matMul.TransposeB
            && Dimension.TryDivExactly(rhsShape[dimInfo.Rn], _nr, out _))
        {
            // 1. Transpose B
            var newRhsPerm = Enumerable.Range(0, rhsShape.Rank).ToArray();
            (newRhsPerm[^2], newRhsPerm[^1]) = (newRhsPerm[^1], newRhsPerm[^2]);
            var newRhs = (Expr)IR.F.Tensors.Transpose(rhs, newRhsPerm);

            // 2. Pack B
            var rN = rhsShape.Rank - 2;
            newRhs = IR.F.Tensors.Pack(newRhs, [_nr], [rN]);
            var output = IR.F.NTT.PackedMatMul(
                lhs,
                newRhs,
                false,
                matMul.OutputDataType);

            // 3. Unpack C
            var cN = output.CheckedShape.Rank - 1;
            output = IR.F.Tensors.Unpack(output, [_nr], [cN]);
            return output;
        }

        return null;
    }
}
