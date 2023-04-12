// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Tensorflow.Operations.Activation;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldSwishPattern1 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(null, "binaryCall", BinaryOp.Mul, IsSigmoid(null, "sigmoidCall", IsWildcard("input")));

    private Expr? GetReplace(Call binaryCall, Call sigmoidCall, Expr input)
    {
        if (binaryCall[Binary.Rhs] == input)
        {
            return IR.F.NN.Swish(input);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldSwishPattern2 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(null, "binaryCall", BinaryOp.Mul, IsWildcard(), IsSigmoid(null, "sigmoidCall", IsWildcard("input")));

    private Expr? GetReplace(Call binaryCall, Call sigmoidCall, Expr input)
    {
        if (binaryCall[Binary.Lhs] == input)
        {
            return IR.F.NN.Swish(input);
        }

        return null;
    }
}
