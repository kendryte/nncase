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
public partial class LowerBinary : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsBinary(
      target_name: "binary",
      _ => true,
      IsWildcard("lhs") with { TypePattern = IsFloat() },
	  IsWildcard("rhs") with { TypePattern = IsFloat() }
	  );

    private Expr? GetReplace(Binary binary, Expr lhs, Expr rhs)
    {
		if (lhs.CheckedShape.Rank != rhs.CheckedShape.Rank)
		{
            return null;
        }
		
        return CPUKernel(binary, lhs, rhs);
    }
}
