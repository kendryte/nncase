using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.K210;

[PatternFunctionalGenerator]
public sealed record Activation() : Op
{
    /// <summary>
    /// Gets input. [n c h w], bf24
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Activation), 0, "input", HasRank(4));

    /// <summary>
    /// Gets act.
    /// Shape: [out_channels, 5]
    /// </summary>
    public static readonly ParameterInfo Act = new(typeof(Activation), 1, "act",HasRank(2));

    /// <summary>
    /// Gets FusedClamp.
    /// </summary>
    public static readonly ParameterInfo Clamps = new(typeof(Activation), 2, "clamp",HasShape(new Shape(2)));
}