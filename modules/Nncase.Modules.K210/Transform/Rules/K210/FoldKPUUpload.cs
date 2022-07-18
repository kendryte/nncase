using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.K210;

[RuleGenerator]
public sealed partial class FoldKPUUpload : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsConv2D(
            null,
            "conv2d",
            PadMode.Constant,
            IsWildcard("input") with { TypePattern = HasFixedShape() },
            IsTensorConst("weights"),
            IsTensorConst("bias"),
            IsTensorConst("strides"),
            IsTensorConst("paddings"),
            new[] { 1, 1 },
            IsTensorConst("groups"),
            IsTensorConst("fusedClamp")) with
        {
            TypePattern = HasFixedShape(),
        };

    private Expr? GetReplace(Expr conv2d, Expr input, Expr weights, Tensor<float> bias, int[] strides, Tensor<int> paddings, int groups, float[] fusedClamp)
    {
        return null;
    }
}