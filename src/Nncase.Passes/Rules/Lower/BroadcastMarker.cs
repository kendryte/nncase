using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;
using static Nncase.Passes.Rules.Lower.BroadcastMarkerHelper;
namespace Nncase.Passes.Rules.Lower;

internal static class BroadcastMarkerHelper
{
    public static bool NotChangeRangeOp(Expr op)
    {
        return op is Squeeze || op is Unsqueeze || op is Reshape;
    }
}

// e.g. matmul(reshape(marker(x))) -> matmul(marker(reshape(marker(x))))
[RuleGenerator]
public partial class BroadcastInputMarker : RewriteRule<Pattern>
{
    override public Pattern Pattern => IsCallWildcard(
        "outer",
        IsWildcard(),
        IsCallWildcard(
            "call",
            IsWildcard(),
            IsRangeOfMarker(
                "marker",
                IsWildcard(),
                IsWildcard())));

    public Expr? GetReplace(Call outer, Call call, Marker marker)
    {
        if (!NotChangeRangeOp(call.Target))
        {
            return null;
        }

        return ReplaceCallFirstParam(outer, marker.With(target: ReplaceCallFirstParam(call, marker)));
    }
}

// e.g. marker(reshape(matmul(x))) -> marker(reshape(marker(matmul(x))))
[RuleGenerator]
public partial class BroadcastOutputMarker : RewriteRule<Pattern>
{
    override public Pattern Pattern => IsRangeOfMarker(
        "marker",
        IsCallWildcard("input", IsWildcard(), IsCallWildcard(null, IsWildcard())),
        IsWildcard());

    public Expr? GetReplace(Call input, Marker marker)
    {
        if (!NotChangeRangeOp(input.Target))
        {
            return null;
        }

        return ReplaceCallFirstParam(input, marker.With(target: input.Arguments[0]));
    }
}
