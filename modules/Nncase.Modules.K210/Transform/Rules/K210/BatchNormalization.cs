// using System;
// using System.Collections.Generic;
// using System.Linq;
// using System.Text;
// using System.Threading.Tasks;
// using Nncase.IR;
// using Nncase.IR.F;
// using Nncase.IR.K210;
// using Nncase.PatternMatch;
// using Nncase.Utilities;
// using static Nncase.IR.TypePatternUtility;
// using static Nncase.PatternMatch.F.NN;
// using static Nncase.PatternMatch.Utility;
//
// namespace Nncase.Transform.Rules.K210;
// [RuleGenerator]
// public sealed partial class BatchNormalization : IRewriteRule
// {
//     public IPattern Pattern { get; } = IsBatchNormalization(
//     )
//     public Expr? GetReplace(IMatchResult result, RunPassOptions options)
//     {
//         throw new NotImplementedException();
//     }
// }