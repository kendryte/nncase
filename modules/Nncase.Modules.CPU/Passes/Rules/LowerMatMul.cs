// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.PatternMatch;

using static Nncase.IR.F.CPU;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules;

[RuleGenerator]
public partial class LowerMatMul : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsMatMul(
      target_name: "matmul",
      _ => true,
      IsWildcard("inputA") with { TypePattern = IsFloat() },
      IsWildcard("inputB") with { TypePattern = IsFloat() });

    private Expr GetReplace(MatMul matmul, Expr inputA, Expr inputB)
    {
        return CPUKernel(matmul, inputA, inputB);
    }
}
