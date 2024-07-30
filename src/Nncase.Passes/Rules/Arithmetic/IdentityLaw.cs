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
/// neg(neg(x)) = x.
/// </summary>
[RuleGenerator]
public sealed partial class DoubleOppositeIsEmpty : IRewriteRule
{
    public IPattern Pattern { get; } = -(-IsWildcard("x"));

    private Expr? GetReplace(Expr x) => x;
}

/// <summary>
/// x - y = x + neg(y).
/// </summary>
[RuleGenerator]
public sealed partial class SubIsAddOpposite : IRewriteRule
{
    public IPattern Pattern { get; } = IsWildcard("x") - IsWildcard("y");

    private Expr? GetReplace(Expr x, Expr y) => x + (-y);
}

/// <summary>
/// x + -x = 0.
/// </summary>
[RuleGenerator]
public sealed partial class XAddNegX : IRewriteRule
{
    public XAddNegX()
    {
        var x = IsWildcard("x");
        Pattern = x + (-x);
    }

    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr x) => Tensor.FromBytes(x.CheckedDataType, new byte[x.CheckedDataType.SizeInBytes], Array.Empty<int>());
}

/// <summary>
/// -0 = 0.
/// </summary>
[RuleGenerator]
public sealed partial class ZeroIsNegZero : IRewriteRule
{
    public ZeroIsNegZero()
    {
        var zero = IsConst("zero", (Const c) =>
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

        Pattern = zero;
    }

    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr zero) => -zero;
}

/// <summary>
/// x + 0 = x.
/// </summary>
[RuleGenerator]
public sealed partial class XIsXAdd0 : IRewriteRule
{
    public XIsXAdd0()
    {
        var x = IsWildcard("x", e => e is not (IR.Tuple or Op));
        Pattern = x;
    }

    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr x) => x + 0;
}

/// <summary>
/// x = x + 0.
/// </summary>
[RuleGenerator]
public sealed partial class XAdd0 : IRewriteRule
{
    public XAdd0()
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

        Pattern = x + 0;
    }

    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr x) => x;
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
