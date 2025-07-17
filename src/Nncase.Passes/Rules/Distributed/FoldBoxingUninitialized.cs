// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.Distributed;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Buffers;
using static Nncase.PatternMatch.F.Distributed;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules;

[RuleGenerator]
public partial class FoldBoxingUninitialized : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsBoxing(
        "boxing",
        "caller",
        _ => true,
        IsUninitialized(
          "target",
          _ => true,
          IsShape("shape")));

    private Expr? GetReplace(Call caller, Uninitialized target, Shape shape)
    {
        var callerType = (DistributedType)caller.CheckedType;
        return IR.F.Buffer.Uninitialized(
            target.DType,
            target.MemoryLocation,
            shape,
            callerType.AxisPolicies,
            callerType.Placement);
    }
}
