using System;
using System.IO;
using Nncase.IR;
using F = Nncase.IR.F;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using Binary = Nncase.IR.Math.Binary;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// int32 * int64 -> int64 * int64
/// </summary>
[RuleGenerator]
public partial class IntegralPromotion : RewriteRule<OrPattern>
{
    private static bool NeedPromotion(Binary bn)
    {
        return bn.BinaryOp switch
        {
            BinaryOp.Add => true,
            BinaryOp.Sub => true,
            BinaryOp.Mul => true,
            BinaryOp.Div => true,
            BinaryOp.Max => true,
            BinaryOp.Min => true,
            _ => false
        };
    }

    /// <inheritdoc/>
    public override OrPattern Pattern { get; } =
        IsAlt(
            IsBinary(
                "bn", NeedPromotion,
                IsWildcard("lhs") with {TypePattern = HasDataType(DataTypes.Int32)},
                IsWildcard("rhs") with {TypePattern = HasDataType(DataTypes.Int64)}),
            IsBinary("bn", NeedPromotion,
                IsWildcard("lhs") with {TypePattern = HasDataType(DataTypes.Int64)},
                IsWildcard("rhs") with {TypePattern = HasDataType(DataTypes.Int32)}));

    Expr GetReplace(Binary bn, Expr lhs, Expr rhs)
    {
        if (lhs.CheckedDataType == DataTypes.Int32 && rhs.CheckedDataType == DataTypes.Int64)
        {
            lhs = F.Tensors.Cast(lhs, DataTypes.Int64);
        }
        else if (lhs.CheckedDataType == DataTypes.Int64 && rhs.CheckedDataType == DataTypes.Int32)
        {
            rhs = F.Tensors.Cast(rhs, DataTypes.Int64);
        }
        else
        {
            throw new InvalidDataException("IntegralPromotion lhs and rhs should be int32 and int64");
        }

        return F.Math.Binary(bn.BinaryOp, lhs, rhs);
    }
}