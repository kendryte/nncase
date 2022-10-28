// using System;
// using System.Collections.Generic;
// using System.Linq;
// using System.Text;
// using System.Threading.Tasks;
// using Nncase.Evaluator;
// using Nncase.IR;
// using Nncase.IR.F;
// using Nncase.IR.K210;
// using Nncase.IR.Math;
// using static Nncase.PatternMatch.F.K210;
// using Nncase.PatternMatch;
// using Nncase.Utilities;
// using Tensorflow.Keras;
// using static Nncase.IR.TypePatternUtility;
// using static Nncase.PatternMatch.F.Math;
// using static Nncase.PatternMatch.F.NN;
// using static Nncase.PatternMatch.Utility;
// using Math = System.Math;
//
// namespace Nncase.Transform.Rules;
// [RuleGenerator]
// public sealed partial class FuseKPUConv2D : IRewriteRule
// {
//     /// <inheritdoc/>
//     public IPattern Pattern { get; } = IsKPUConv2D(
//         null,
//         "conv2d_call",
//         op => true,
//         IsRangeOfMarker(IsWildcard("input")with { TypePattern = HasFixedShape() },
//             IsConst("input_range")),
//         IsTensorConst("weights"),
//         IsTensorConst("batchNorms"),
//         IsTensorConst("outputquantparam"),
//         IsTensorConst(),
//         IsTensorConst(),
//         IsTensorConst());
//
//     private Expr? GetReplace(Call upload_call, Expr input, Expr weights, Expr batchNorms, Expr outputquantparam)
//     {
//         return null;
//     }
// }