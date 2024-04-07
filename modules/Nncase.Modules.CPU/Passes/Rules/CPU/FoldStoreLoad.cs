// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.CPU;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CPU;

[RuleGenerator]
public sealed partial class FoldStoreLoad : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsLoad(
            _ => true,
            IsStore(
                _ => true,
                input: IsWildcard("input")));

    public Expr? GetReplace(Expr input)
    {
        return input;
    }
}
