using System;
using Nncase.IR;
using F = Nncase.IR.F;
using Nncase.PatternMatch;
using Tensorflow;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using Binary = Nncase.IR.Math.Binary;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// int32 * int64 -> int64 * int64
/// </summary>
[RuleGenerator]
public partial class IntegralPromotion : RewriteRule<CallPattern>
{
    private static bool NeedPromotion(Binary bn)
    {
        return bn.BinaryOp switch
        {
            BinaryOp.Add => true,
            BinaryOp.Sub => true,
            BinaryOp.Mul => true,
            BinaryOp.Div => true,
            _ => false
        };
    }
    
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = 
        IsBinary(NeedPromotion,
            IsWildcard("lhs") with { TypePattern = IsDataType(DataTypes.Int32)}, 
            IsWildcard("rhs") with { TypePattern = IsDataType(DataTypes.Int64)}) | 
        IsBinary("bn", NeedPromotion,
            IsWildcard("lhs") with { TypePattern = IsDataType(DataTypes.Int64)}, 
            IsWildcard("rhs") with { TypePattern = IsDataType(DataTypes.Int32)});

    Expr GetReplace(Binary bn, Expr lhs, Expr rhs)
    {
        if (lhs.CheckedDataType == DataTypes.Int32 && rhs.CheckedDataType == DataTypes.Int64)
        {
            lhs = F.Tensors.Cast(lhs, DataTypes.Int64);
        }
        else if (rhs.CheckedDataType == DataTypes.Int64 && rhs.CheckedDataType == DataTypes.Int32)
        {
            rhs = F.Tensors.Cast(rhs, DataTypes.Int64);
        }
        else
        {
            throw new InvalidArgumentError("IntegralPromotion lhs and rhs should be int32 and int64");
        }

        return F.Math.Binary(bn.BinaryOp, lhs, rhs);
    }
}