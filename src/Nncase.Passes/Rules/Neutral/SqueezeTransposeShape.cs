// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using DryIoc;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Tensorflow;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Squeeze transpose shape.
/// </summary>
[RuleGenerator]
public sealed partial class Squeeze5DTranspose : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsTranspose(
        "transpose",
        "call",
        IsWildcard("input") with { TypePattern = HasFixedShape() & HasRank(x => x > 4, "more than 4D need to squeeze") },
        IsWildcard("perm"));

    private Expr? GetReplace(Expr call, Expr input, int[] perm)
    {
        var inputShape = input.CheckedShape.ToValueList();

        var shape1 = inputShape.GetRange(0, 3);
        shape1.Add(inputShape[3] * inputShape[4]);
        var perm1 = ((int[])perm.Clone()).RemoveAt(perm.IndexOf(4));
        var tp1 = Transpose(Reshape(input, shape1.ToArray()), perm1);
        var shape2 = perm1.Select(p => shape1[p]).ToArray();

        int[] perm2;
        int[] shape3;
        switch (perm1.IndexOf(3))
        {
            case 0:
                switch (perm.IndexOf(4))
                {
                    case 0:
                        perm2 = new[] { 1, 0, 2, 3 };
                        shape3 = new[] { inputShape[3], inputShape[4], shape2[1], shape2[2] * shape2[3] };
                        break;
                    case 1:
                        perm2 = new[] { 0, 1, 2, 3 };
                        shape3 = new[] { inputShape[3], inputShape[4], shape2[1], shape2[2] * shape2[3] };
                        break;
                    case 2:
                        perm2 = new[] { 0, 2, 1, 3 };
                        shape3 = new[] { inputShape[3], inputShape[4], shape2[1], shape2[2] * shape2[3] };
                        break;
                    case 3:
                        perm2 = new[] { 0, 2, 1, 3 };
                        shape3 = new[] { inputShape[3], inputShape[4], shape2[1] * shape2[2], shape2[3] };
                        break;
                    default:
                        perm2 = new[] { 0, 2, 3, 1 };
                        shape3 = new[] { inputShape[3], inputShape[4], shape2[1] * shape2[2], shape2[3] };
                        break;
                }

                break;
            case 1:
                {
                    switch (perm.IndexOf(4))
                    {
                        case 0:
                            perm2 = new[] { 2, 0, 1, 3 };
                            shape3 = new[] { shape2[0], inputShape[3], inputShape[4], shape2[2] * shape2[3] };
                            break;
                        case 1:
                            perm2 = new[] { 0, 2, 1, 3 };
                            shape3 = new[] { shape2[0], inputShape[3], inputShape[4], shape2[2] * shape2[3] };
                            break;
                        case 2:
                            perm2 = new[] { 0, 1, 2, 3 };
                            shape3 = new[] { shape2[0], inputShape[3], inputShape[4], shape2[2] * shape2[3] };
                            break;
                        case 3:
                            perm2 = new[] { 0, 2, 1, 3 };
                            shape3 = new[] { shape2[0] * inputShape[3], inputShape[4], shape2[2], shape2[3] };
                            break;
                        default:
                            perm2 = new[] { 0, 1, 3, 2 };
                            shape3 = new[] { shape2[0], inputShape[3], inputShape[4], shape2[2] * shape2[3] };
                            break;
                    }

                    break;
                }

            case 2:
                {
                    switch (perm.IndexOf(4))
                    {
                        case 0:
                            perm2 = new[] { 2, 0, 1, 3 };
                            shape3 = new[] { shape2[0] * shape2[1], inputShape[3], inputShape[4], shape2[3] };
                            break;
                        case 1:
                            perm2 = new[] { 0, 2, 1, 3 };
                            shape3 = new[] { shape2[0], shape2[1] * inputShape[3], inputShape[4], shape2[3] };
                            break;
                        case 2:
                            perm2 = new[] { 0, 2, 1, 3 };
                            shape3 = new[] { shape2[0] * shape2[1], inputShape[3], inputShape[4], shape2[3] };
                            break;
                        case 3:
                            perm2 = new[] { 0, 1, 2, 3 };
                            shape3 = new[] { shape2[0] * shape2[1], inputShape[3], inputShape[4], shape2[3] };
                            break;
                        default:
                            perm2 = new[] { 0, 1, 3, 2 };
                            shape3 = new[] { shape2[0] * shape2[1], inputShape[3], inputShape[4], shape2[3] };
                            break;
                    }

                    break;
                }

            case 3:
                {
                    switch (perm.IndexOf(4))
                    {
                        case 0:
                            perm2 = new[] { 3, 0, 1, 2 };
                            shape3 = new[] { shape2[0] * shape2[1], shape2[2], inputShape[3], inputShape[4] };
                            break;
                        case 1:
                            perm2 = new[] { 0, 3, 1, 2 };
                            shape3 = new[] { shape2[0], shape2[1] * shape2[2], inputShape[3], inputShape[4] };
                            break;
                        case 2:
                            perm2 = new[] { 0, 3, 1, 2 };
                            shape3 = new[] { shape2[0] * shape2[1], shape2[2], inputShape[3], inputShape[4] };
                            break;
                        case 3:
                            perm2 = new[] { 0, 1, 3, 2 };
                            shape3 = new[] { shape2[0] * shape2[1], shape2[2], inputShape[3], inputShape[4] };
                            break;
                        default:
                            perm2 = new[] { 0, 1, 2, 3 };
                            shape3 = new[] { shape2[0] * shape2[1], shape2[2], inputShape[3], inputShape[4] };
                            break;
                    }

                    break;
                }

            default:
                throw new NotSupportedException("Not Supported perm!");
        }

        return Reshape(Transpose(Reshape(tp1, shape3), perm2), call.CheckedShape);
    }
}
