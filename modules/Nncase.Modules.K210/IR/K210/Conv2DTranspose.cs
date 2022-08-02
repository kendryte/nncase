using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.K210;

[PatternFunctionalGenerator]
public sealed record class Conv2DTranspose(bool IsDepthwise, KPUFilterType FilterType, KPUPoolType PoolType, Tensor<float> Bias, ValueRange<float> FusedClamp) : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Conv2DTranspose), 0, "input", HasRank(4));

    /// <summary>
    /// Gets Weights.
    /// </summary>
    public static readonly ParameterInfo Weights = new(typeof(Conv2DTranspose), 1, "weights", HasRank(4));
}