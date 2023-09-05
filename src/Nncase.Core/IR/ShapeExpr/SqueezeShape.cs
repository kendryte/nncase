using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.ShapeExpr;

/// <summary>
/// Squeeze expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class SqueezeShape : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(SqueezeShape), 0, "input");

    /// <summary>
    /// Gets dimension.
    /// </summary>
    public static readonly ParameterInfo Dim = new(typeof(SqueezeShape), 1, "dim", HasRank(1) & IsIntegral());
}
