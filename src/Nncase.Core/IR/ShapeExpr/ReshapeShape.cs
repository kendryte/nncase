using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.ShapeExpr;

/// <summary>
/// Reshape expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class ReshapeShape : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(ReshapeShape), 0, "input");

    /// <summary>
    /// Gets shape.
    /// </summary>
    public static readonly ParameterInfo Shape = new(typeof(ReshapeShape), 1, "shape", HasRank(1));
}
