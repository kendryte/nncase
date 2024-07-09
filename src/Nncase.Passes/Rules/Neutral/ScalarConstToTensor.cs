// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.Passes.Utility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class BianryScalarConstToTensor : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsBinary(
        "bn",
        "bnCall",
        _ => true,
        IsWildcard("lhs"),
        IsWildcard("rhs"));

    private Expr? GetReplace(Call bnCall, Expr lhs, Expr rhs)
    {
        if (bnCall.Arguments.AsValueEnumerable().Any(a => a is TensorConst { Value: Tensor { Shape.IsScalar: true } }))
        {
            var arguments = bnCall.Arguments.AsValueEnumerable().Select(e => e switch { TensorConst { Value: Tensor { Shape.IsScalar: true } } tc => Const.FromTensor(Tensor.FromBytes(tc.CheckedDataType, tc.Value.BytesBuffer.ToArray(), new[] { 1 })), _ => e }).ToArray();
            return bnCall.With(arguments: arguments);
        }

        return null;
    }
}
