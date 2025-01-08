// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class ReshapeExpand : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsExpand(
        null,
        "expand",
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("shape") with { TypePattern = HasShape(shape => shape[0].FixedValue > 4, string.Empty) });

    private Expr? GetReplace(Expr expand, Expr input, long[] shape, RunPassContext context)
    {
        var newOrginShape = new List<long>();
        var newShape = new List<long>();
        long dim = 1;
        for (var i = 0; i < shape.Length; i++)
        {
            var lhs = input.CheckedShape[i].FixedValue;
            var rhs = shape[i];
            if (lhs == rhs)
            {
                dim *= lhs;
            }
            else
            {
                newOrginShape.Add(dim);
                newShape.Add(dim);
                dim = 1;
                newOrginShape.Add(lhs);
                newShape.Add(rhs);
            }
        }

        if (dim != 1)
        {
            newOrginShape.Add(dim);
            newShape.Add(dim);
        }

        if (newShape.Count == shape.Length) // No 1 exists
        {
            return null;
        }

        var newInput = IR.F.Tensors.Reshape(input, newOrginShape.ToArray());
        var end = IR.F.Tensors.Reshape(IR.F.Tensors.Expand(newInput, newShape.ToArray()).InheritMetaData(expand), shape);
        return end;
    }
}
