// using System;
// using System.Collections.Generic;
// using System.Linq;
// using System.Text;
// using System.Threading.Tasks;
// using Nncase.IR;
// using Nncase.IR.F;
// using Nncase.IR.K210;
// using Nncase.IR.NN;
// using Nncase.PatternMatch;
// using Nncase.Utilities;
// using static Nncase.IR.TypePatternUtility;
// using static Nncase.PatternMatch.F.NN;
// using static Nncase.PatternMatch.Utility;
//
// namespace Nncase.Transform.Rules;
// [RuleGenerator]
// public sealed partial class Conv2dTransposeTransform : RewriteRule<Pattern>
// {
//     public override IPattern Pattern { get; } = 
//         IsConv2DTranspose("conv2dTranspose", _ => true,
//         IsWildcard("input"),
//         IsTensorConst("weights"),
//         IsTensorConst("bias"),
//         IsWildcard("outShape"),
//         IsWildcard("stride"),
//         IsWildcard("padding"),
//         IsWildcard("outPadding"), 
//         IsWildcard("dilation"),
//         IsWildcard("groups"),
//         IsWildcard("fusedClamp"));
//
//     private Expr? GetReplace(Conv2DTranspose conv2dTranspose, Expr input, Expr weights, TensorConst bias, Expr outShape, Expr stride, Expr padding, 
//         int[] outPadding, Expr dilation, Expr groups, Expr fusedClamp)
//     {
//         // return IR.NN.Conv2DTranspose(input, weights, None.Default, KPUUtility.GetDefaultConvActParam(weights, bias),
//         //     outShape, padding, outPadding, stride, dilation, groups, fusedClamp);
//         if (outPadding.Sum() != 0)
//         {
//             return null;
//         }
//         var act = KPUUtility.GetDefaultConvActParam(weights, bias);
//         return IR.F.K210.Conv2DTranspose(input, weights,  None.Default,
//                 act, outShape, padding,outPadding,
//                 stride, dilation, groups, fusedClamp);
//     }
// }