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
public sealed partial class PackedMatMulUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsPackedMatMul(
            "matMul",
            "caller",
            _ => true,
            PatternMatch.F.Tensors.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("lhs")),
            IsTensorConst("rhs"));

    private Expr? GetReplace(Unpack unpack, PackedMatMul matMul, Call caller, Call callee, Expr lhs, Tensor rhs)
    {
        var lhsShape = lhs.CheckedShape;
        var dimInfo = matMul.GetDimInfo(lhsShape.Rank, rhs.Rank);
        (var lhsPackKind, var rhsPackKind) = matMul.GetPackKind(lhsShape.Rank, rhs.Rank);
        if (lhsPackKind == PackedMatMul.PackKind.None && rhsPackKind == PackedMatMul.PackKind.N && unpack.Axes == [dimInfo.Lk])
        {
            // If the unpack is on K, we can pack the rhs with KN
            var rhsLanes = ((VectorType)rhs.ElementType).Lanes.ToArray();
            var unpackedRhs = IR.F.Tensors.Unpack(rhs, rhsLanes, [rhs.Rank - 1]);
            rhsLanes = [.. unpack.Lanes, .. rhsLanes];
            IRArray<int> rhsPackedAxes = [rhs.Rank - 2, rhs.Rank - 1];
            var packedRhs = IR.F.Tensors.Pack(unpackedRhs, rhsLanes, rhsPackedAxes.ToArray());
            return IR.F.NTT.PackedMatMul(
                lhs,
                packedRhs,
                unpack.Axes,
                rhsPackedAxes,
                matMul.TransposeA,
                matMul.TransposeB,
                matMul.FusedReduce,
                matMul.OutputDataType);
        }

        return null;
    }
}
// // Copyright (c) Canaan Inc. All rights reserved.
// // Licensed under the Apache license. See LICENSE file in the project root for full license information.

// using System;
// using System.Collections.Generic;
// using System.Linq;
// using System.Text;
// using System.Threading.Tasks;
// using DryIoc.ImTools;
// using NetFabric.Hyperlinq;
// using Nncase.IR;
// using Nncase.IR.NN;
// using Nncase.IR.NTT;
// using Nncase.IR.Shapes;
// using Nncase.IR.Tensors;
// using Nncase.PatternMatch;
// using Nncase.Utilities;

// using static Nncase.IR.TypePatternUtility;
// using static Nncase.PatternMatch.F.Math;
// using static Nncase.PatternMatch.F.NTT;
// using static Nncase.PatternMatch.F.Tensors;
// using static Nncase.PatternMatch.Utility;

// namespace Nncase.Passes.Rules.NTT;

// [RuleGenerator]
// public sealed partial class PackedMatMulUnpackPropagation : RewriteRule<Pattern>
// {
//     public override Pattern Pattern { get; } =
//         IsPackedMatMul(
//             "matMul",
//             "caller",
//             _ => true,
//             PatternMatch.F.Tensors.IsUnpack(
//                 "unpack",
//                 "callee",
//                 _ => true,
//                 IsWildcard("lhs")),
//             IsTensorConst("rhs"));

//     private Expr? GetReplace(Unpack unpack, PackedMatMul matMul, Call caller, Call callee, Expr lhs, Tensor rhs)
//     {
//         var lhsShape = lhs.CheckedShape;
//         var dimInfo = matMul.GetDimInfo(lhsShape.Rank, rhs.Rank);
//         (var lhsPackKind, var rhsPackKind) = matMul.GetPackKind(lhsShape.Rank, rhs.Rank);
//         if (lhsPackKind == PackedMatMul.PackKind.None && rhsPackKind == PackedMatMul.PackKind.N && unpack.Axes == [dimInfo.Lk])
//         {
//             // If the unpack is on K, we can bitcast the lhs to element type.
//             var newDType = ((VectorType)lhs.CheckedTensorType.DType).ElemType;
//             return caller.WithArguments([(PackedMatMul.Lhs, IR.F.Tensors.Bitcast(lhs, newDType))]);
//         }

//         return null;
//     }
// }
