// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Arithmetic;

/// <summary>
/// x * 1 => x.
/// </summary>
[RuleGenerator]
public sealed partial class XMul1 : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsWildcard("x") * 1.0;

    private Expr? GetReplace(Expr x) => x;
}

/// <summary>
/// x + -x => 0.
/// </summary>
[RuleGenerator]
public sealed partial class XNegX : IRewriteRule
{
    public XNegX()
    {
        var x = IsWildcard("x");
        Pattern = x + (-x);
    }

    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr x) => Tensor.FromBytes(x.CheckedDataType, new byte[x.CheckedDataType.SizeInBytes], Array.Empty<int>());
}

/// <summary>
/// x - (x ± 0) or  (x ± 0) - x => x.
/// because of simplify x+0 => x will cause the egraph explosion.
/// </summary>
[RuleGenerator]
public sealed partial class XNegX0 : IRewriteRule
{
    public XNegX0()
    {
        var x = IsWildcard("x");
        var zero = IsConst((Const c) =>
        {
            if (c is TensorConst tc)
            {
                var bf = tc.Value.BytesBuffer;
                bool isZero = true;
                for (int i = 0; i < bf.Length; i++)
                {
                    if (bf[i] != 0)
                    {
                        isZero = false;
                    }
                }

                return isZero;
            }

            return false;
        });

        var y = IsSwappableBinary("b0", "c0", b => b.BinaryOp is BinaryOp.Add or BinaryOp.Sub, x, zero);
        Pattern = IsAlt(-y + x, x - y, -x + y, y - x);
    }

    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr x) => Tensor.FromBytes(x.CheckedDataType, new byte[x.CheckedDataType.SizeInBytes], Array.Empty<int>());
}

/// <summary>
/// x / x => 1.
/// </summary>
[RuleGenerator]
public sealed partial class XDivX : IRewriteRule
{
    /// <summary>
    /// Initializes a new instance of the <see cref="XDivX"/> class.
    /// </summary>
    public XDivX()
    {
        var x = IsWildcard("x") with { TypePattern = IsTensor() };
        Pattern = x / x;
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr x)
    {
        var value = ((Tensor)1).CastTo(x.CheckedDataType);
        return Tensors.ConstantOfShape(Tensors.ShapeOf(x), value);
    }
}
