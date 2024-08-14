// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Decompose instancenorm.
/// </summary>
[RuleGenerator]
public sealed partial class DecomposeInstanceNorm : IRewriteRule
{
  /// <inheritdoc/>
  public IPattern Pattern { get; } =
  IsInstanceNormalization(
    "instanceNorm",
    "call",
    _ => true,
    IsWildcard("input") with { TypePattern = IsFloat() },
    IsTensorConst("scale") with { TypePattern = IsFloat() },
    IsTensorConst("bias") with { TypePattern = IsFloat() },
    IsTensorConst("eps") with { TypePattern = IsFloat() });

  private Expr? GetReplace(Expr input, Call call, InstanceNormalization instanceNorm, TensorConst scale, TensorConst bias, TensorConst eps)
  {
    var axis = Enumerable.Range(2, call.CheckedShape.Rank - 2).ToArray();
    var mean = IR.F.Tensors.ReduceMean(input, axis, 0f, true);
    var sub = IR.F.Math.Sub(input, mean);
    var sigma = IR.F.Tensors.ReduceMean(IR.F.Math.Square(sub), axis, 0f, true);
    var rsigma = IR.F.Math.Rsqrt(IR.F.Math.Add(sigma, eps));
    var newScale = Tensor.FromBytes(scale.CheckedDataType, scale.Value.BytesBuffer.ToArray(), new[] { scale.CheckedShape[0].FixedValue, 1, 1 });
    var newBias = Tensor.FromBytes(bias.CheckedDataType, bias.Value.BytesBuffer.ToArray(), new[] { bias.CheckedShape[0].FixedValue, 1, 1 });
    return IR.F.Math.Add(IR.F.Math.Mul(IR.F.Math.Mul(sub, rsigma), newScale), newBias);
  }
}
