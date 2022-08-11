// using Nncase.IR;
// using Nncase.IR.K210;
// using Nncase.IR.NN;
// using Nncase.PatternMatch;
// namespace Nncase.Transform.Rules.K210;
// using static Nncase.PatternMatch.F.NN;
// using static Nncase.PatternMatch.Utility;
// [RuleGenerator]
// public sealed partial class ToFakeConv2D : RewriteRule<Pattern>
// {
//     /// <inheritdoc/>
//     public override Pattern Pattern { get; } =
//         IsConv2D("conv", "call", _ => true,
//             IsWildcard("input"),
//             IsTensorConst("weights"),
//             IsTensorConst("bias"),
//             IsTensorConst("stride"),
//             IsWildcard("padding"),
//             IsTensorConst("dilation"),
//             IsTensorConst("groups"),
//             IsWildcard("fusedClamp"));
//
//     private Expr? GetReplace(Conv2D conv, Expr input, Expr weights, TensorConst bias, Expr stride, Expr padding,
//         Expr dilation, Expr groups, Expr fusedClamp)
//     {
//         return FakeKPUConv2D(
//             IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), IR.F.Math.RangeOfMarker(weights, IR.F.Math.RangeOf(weights)),
//             None.Default,
//             KPUUtility.GetFakeConvActParam(weights, bias), padding, stride, dilation, groups);
//     }
// }