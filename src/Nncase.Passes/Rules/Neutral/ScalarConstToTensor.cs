// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.Passes.Utility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class ScalarConstToTensor : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsCallWildcard(
        "call",
        IsAlt(IsOp<IR.Math.Binary>(), IsOp<IR.Math.Unary>(), IsOp<IR.Tensors.Where>(), IsOp<IR.Math.Compare>()));

    private Expr? GetReplace(Call call)
    {
        if (call.Arguments.AsValueEnumerable().Any(a => a is TensorConst { Value: Tensor { Shape.IsScalar: true } }))
        {
            var arguments = call.Arguments.AsValueEnumerable().Select(e => e switch { TensorConst { Value: Tensor { Shape.IsScalar: true } } tc => Const.FromTensor(Tensor.FromBytes(tc.CheckedDataType, tc.Value.BytesBuffer.ToArray(), new[] { 1 })), _ => e }).ToArray();
            return call.With(arguments: arguments);
        }

        return null;
    }
}
