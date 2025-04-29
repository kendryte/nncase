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
using static Nncase.IR.F.Math;
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
        IsWildcard("input") with { TypePattern = HasFixedShape() & HasRank(x => x == 5, "more than 4D need to squeeze") },
        IsWildcard("perm"));

    private Expr? GetReplace(Expr call, Expr input, int[] perm)
    {
        var inputShape = input.CheckedShape.ToValueArray();

        var shape1 = inputShape[0..3].ToList();
        shape1.Add(inputShape[3] * inputShape[4]);
        var perm1 = ((int[])perm.Clone()).RemoveAt(perm.IndexOf(4));
        var tp1 = Transpose(Reshape(input, shape1.ToArray()), perm1);
        var shape2 = perm1.Select(p => shape1[p]).ToArray();

        int[] perm2;
        long[] shape3;
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

        return Reshape(Transpose(Reshape(tp1, shape3), perm2).With(metadata: call.Metadata), call.CheckedShape);
    }
}

[RuleGenerator]
public sealed partial class SqueezeTransposeShape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsTranspose(
        "transpose",
        "call",
        IsWildcard("input") with { TypePattern = HasFixedShape() & HasRank(x => x > 4, "more than 4D need to squeeze") },
        IsWildcard("perm"));

    private Tuple<bool, List<int>, List<long>> SqueezeTranspose(long[] oldShape, List<int> oldAxis)
    {
        if (oldShape.Length <= 4)
        {
            return new Tuple<bool, List<int>, List<long>>(false, oldAxis, oldShape.ToList());
        }

        var newAxis = new List<int>(oldAxis);
        var newShape = new List<long>(oldShape);
        int squeezeTimes = oldShape.Length - 4;

        var foldIndexCouple = new List<Tuple<int, int>>();
        for (int i = oldShape.Length - 1; i > 0; i--)
        {
            if (oldAxis[i - 1] + 1 == oldAxis[i])
            {
                foldIndexCouple.Add(new Tuple<int, int>(i - 1, i));
            }
        }

        if (foldIndexCouple.Count < squeezeTimes)
        {
            return new Tuple<bool, List<int>, List<long>>(false, newAxis, newShape);
        }

        while (squeezeTimes > 0 && foldIndexCouple.Count > 0)
        {
            Tuple<int, int> it = foldIndexCouple[0];
            int front = it.Item1;
            int back = it.Item2;
            newShape[oldAxis[front]] *= newShape[oldAxis[back]];
            newShape.RemoveAt(oldAxis[back]);
            newAxis.RemoveAt(back);
            foldIndexCouple.RemoveAt(0);
            squeezeTimes--;
        }

        // fix axis
        for (int i = 0, j = 0; j < 4; i++)
        {
            int index = newAxis.IndexOf(i);
            if (index != -1)
            {
                newAxis[index] = j;
                j++;
            }
        }

        return new Tuple<bool, List<int>, List<long>>(true, newAxis, newShape);
    }

    private Expr? GetReplace(Expr input, int[] perm, Expr call)
    {
        var inputShape = input.CheckedShape;
        var (result, new_perm, new_shape) = SqueezeTranspose(inputShape.ToValueArray(), perm.ToList());
        if (!result)
        {
            return null;
        }

        var newOutputShape = new long[perm.Length];
        for (int i = 0; i < perm.Length; i++)
        {
            newOutputShape[i] = inputShape[perm[i]].FixedValue;
        }

        return Reshape(Transpose(Reshape(input, new_shape.ToArray()), new_perm.ToArray()).With(metadata: call.Metadata), newOutputShape);
    }
}

[RuleGenerator]
public sealed partial class SqueezeBinaryShape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsBinary("binary", "binaryCall", x => true, IsWildcard("lhs") with { TypePattern = HasFixedShape() }, IsWildcard("rhs") with { TypePattern = HasFixedShape() });

    /// <summary>
    /// Squeeze input shape.
    /// </summary>
    /// <param name="a"> left input shape.</param>
    /// <param name="b"> right input shape.</param>
    /// <returns> Squeeze flag, new lhs, new rhs. </returns>
    public (bool SqueezeOrNot, IReadOnlyList<long> NewAShape, IReadOnlyList<long> NewBShape) SqueezeInputShape(long[] a, long[] b)
    {
        var aSize = a.Length;
        var bSize = b.Length;

        var squeezeTimes = Math.Max(
            aSize > 4 ? aSize - 4 : 0,
            bSize > 4 ? bSize - 4 : 0);

        if (squeezeTimes <= 0)
        {
            return (false, a, b);
        }

        var newA = a.ToList();
        var newB = b.ToList();

        if (aSize == bSize)
        {
            if (a.SequenceEqual(b))
            {
                newA = SqueezeShape(a);
                newB = SqueezeShape(b);
            }
            else
            {
                var canFold = Enumerable.Repeat(true, aSize).ToArray();
                var foldIndexCouples = new List<(int, int)>();

                for (int i = 0; i < aSize; i++)
                {
                    if (a[i] != b[i])
                    {
                        canFold[i] = false;
                    }
                }

                for (int i = aSize - 1; i > 0; i--)
                {
                    if (canFold[i] && canFold[i - 1])
                    {
                        foldIndexCouples.Add((i - 1, i));
                    }
                }

                while (squeezeTimes > 0 && foldIndexCouples.Count > 0)
                {
                    var (front, back) = foldIndexCouples[0];
                    newA[front] *= newA[back];
                    newB[front] *= newB[back];

                    newA.RemoveAt(back);
                    newB.RemoveAt(back);

                    foldIndexCouples.RemoveAt(0);
                    squeezeTimes--;
                }

                for (int i = newA.Count - 1, count = newA.Count - 5; i >= 0 && count >= 0; i--)
                {
                    if (newA[i] * newB[i] == 1)
                    {
                        newA.RemoveAt(i);
                        newB.RemoveAt(i);
                        count--;
                    }
                }

                if (newA.Count > 4)
                {
                    return (false, newA, newB);
                }
            }
        }
        else
        {
            if (aSize != 1)
            {
                newA = SqueezeShape(a);
            }

            if (bSize != 1)
            {
                newB = SqueezeShape(b);
            }
        }

        return (true, newA, newB);
    }

    private static List<long> SqueezeShape(long[] shape)
    {
        var newShape = new List<long> { 1, 1, 1, 1 };

        for (int i = shape.Length - 1, k = 3; i >= 0; i--)
        {
            newShape[k] *= shape[i];
            if (k > 0)
            {
                k--;
            }
        }

        return newShape;
    }

    private static long[] GetOutputShape(long[] a, long[] b)
    {
        if (a.Length == 1)
        {
            return b;
        }

        if (b.Length == 1)
        {
            return a;
        }

        var outputShape = a;
        for (int i = 0; i < a.Length; i++)
        {
            outputShape[i] = Math.Max(a[i], b[i]);
        }

        return outputShape;
    }

    private Expr? GetReplace(Binary binary, Call binaryCall, Expr lhs, Expr rhs)
    {
        var lhsShape = (RankedShape)lhs.CheckedShape;
        var rhsShape = (RankedShape)rhs.CheckedShape;
        var lShape = lhsShape.Count == 0 ? new RankedShape(1) : lhsShape;
        var rShape = rhsShape.Count == 0 ? new RankedShape(1) : rhsShape;
        var (result, newLShape, newRShape) = SqueezeInputShape(lShape.ToValueArray(), rShape.ToValueArray());
        if (!result)
        {
            return null;
        }

        var outputShape = GetOutputShape(lShape.ToValueArray(), rShape.ToValueArray());

        return Reshape(Binary(binary.BinaryOp, Reshape(lhs, newLShape.ToArray()).With(metadata: lhs.Metadata), Reshape(rhs, newRShape.ToArray()).With(metadata: rhs.Metadata)).With(metadata: binaryCall.Metadata), outputShape.ToArray());
    }
}
