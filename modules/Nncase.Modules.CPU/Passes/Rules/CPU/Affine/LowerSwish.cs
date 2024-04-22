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
public partial class LowerSwish : RewriteRule<Pattern>
{
    private int _count;

    /// <inheritdoc/>
    public override Pattern Pattern { get; } = PatternMatch.F.NN.IsSwish(
      "swish",
      "call",
      IsWildcard("input") with { TypePattern = HasFixedShape() },
      IsTensorConst("beta") with { TypePattern = IsFloat() & (IsScalar() | HasShape(s => s.Rank == 1 && s[0].FixedValue == 1, "scalar")) });

    private Expr GetReplace(Call call, Expr input, float beta)
    {
        var rank = input.CheckedShape.Rank;
        var bufferType = input.CheckedType switch
        {
            TensorType t => t,
            DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt),
            _ => throw new ArgumentOutOfRangeException(nameof(input)),
        };
        return IR.F.Affine.Grid(CPUTarget.Kind)
            .Read(input, AffineMap.Identity(rank), out var inTile)
            .Write(TIR.T.CreateBuffer(bufferType, TIR.MemoryLocation.Data, out _, $"swish_{_count++}"), AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.CPU.Swish(inTile, outTile, beta))
            .Build();
    }
}
