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
    public LowerSwish(string moduleKind = CPUTarget.Kind)
    {
        ModuleKind = moduleKind;
    }

    public string ModuleKind { get; }

    /// <inheritdoc/>
    public override Pattern Pattern { get; } = PatternMatch.F.NN.IsSwish(
      "swish",
      "call",
      IsWildcard("input") with { TypePattern = HasFixedShape() },
      IsTensorConst("beta") with { TypePattern = IsFloat() & (IsScalar() | HasShape(s => s.Rank == 1 && s[0].FixedValue == 1, "scalar")) });

    private Expr GetReplace(Call call, Expr input, float beta)
    {
        var outBuffer = input.CheckedType switch
        {
            TensorType t => IR.F.Buffer.Uninitialized(t.DType, TIR.MemoryLocation.Data, t.Shape.ToValueArray()),
            DistributedType dt => IR.F.Buffer.Uninitialized(dt.TensorType.DType, TIR.MemoryLocation.Data, dt.TensorType.Shape.ToValueArray(), dt.NdSBP, dt.Placement),
            _ => throw new ArgumentOutOfRangeException(nameof(input)),
        };
        var rank = input.CheckedShape.Rank;
        return IR.F.Affine.Grid(ModuleKind)
            .Read(input, AffineMap.Identity(rank), out var inTile)
            .Write(outBuffer, AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.CPU.Swish(inTile, outTile, beta))
            .Build();
    }
}
