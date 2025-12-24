// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Reactive;
using Nncase.IR;

namespace Nncase.Layouts;

public static class LayoutUtilities
{
    public static CollectValue Flatten(RecursiveValue t)
    {
        switch (t)
        {
            case IntValue i:
                return new CollectValue(new[] { i });
            case CollectValue tp:
                return new CollectValue(tp.Elements.SelectMany(x => Flatten(x).Elements));
            default:
                throw new NotSupportedException();
        }
    }

    public static Layout Unflatten(Layout layout, RecursiveValue profile)
    {
        return new Layout(Unflatten(layout.Shape, profile), Unflatten(layout.Stride, profile));
    }

    public static RecursiveValue Unflatten(RecursiveValue value, RecursiveValue profile)
    {
        CollectValue UnflattenImpl(RecursiveValue value, RecursiveValue profile)
        {
            switch (value, profile)
            {
                case (_, CollectValue tprofile):
                    var result = new List<RecursiveValue>();
                    RecursiveValue remaining = value;
                    foreach (var p in tprofile)
                    {
                        if (UnflattenImpl(remaining, p) is CollectValue { Count: 2 } subResult &&
                            subResult.Elements[0] is RecursiveValue unflattened &&
                            subResult.Elements[1] is CollectValue remainder)
                        {
                            switch (unflattened)
                            {
                                case CollectValue ct:
                                    result.AddRange(ct.Elements);
                                    break;
                                case IntValue iv:
                                    result.Add(unflattened);
                                    break;
                            }

                            remaining = remainder;
                        }
                        else
                        {
                            throw new NotSupportedException();
                        }
                    }

                    return new CollectValue([new CollectValue(result), remaining]);
                case (CollectValue tuple, _):
                    return new CollectValue([tuple[0], tuple[1..]]);
                case (IntValue intValue, _):
                    return new CollectValue([intValue, new CollectValue([])]);
                default:
                    throw new NotSupportedException();
            }
        }

        var result = UnflattenImpl(value, profile);
        if (result[1] is not CollectValue { Count: 0 })
        {
            throw new ArgumentException("profile not suitable", nameof(profile));
        }

        return result[0];
    }

    public static int Signum(RecursiveValue a)
    {
        switch (a)
        {
            case IntValue i:
                var val = i.Value;
                return (val > 0 ? 1 : 0) - (val < 0 ? 1 : 0);
            default:
                throw new NotSupportedException();
        }
    }

    public static long Product(RecursiveValue a)
    {
        switch (a)
        {
            case CollectValue tuple:
                return tuple.Elements.Aggregate(1L, (val, elem) => val * Product(elem));
            case IntValue i:
                return i.Value;
            default:
                throw new NotSupportedException();
        }
    }

    public static long InnerProduct(RecursiveValue a, RecursiveValue b)
    {
        switch (a, b)
        {
            case (CollectValue ta, CollectValue tb):
                if (ta.Count != tb.Count)
                {
                    throw new NotSupportedException();
                }

                return ta.Elements.Zip(tb.Elements, InnerProduct).Sum();
            case (IntValue ai, IntValue bi):
                return ai.Value * bi.Value;
            default:
                throw new NotSupportedException();
        }
    }

    public static long TupleMax(RecursiveValue a)
    {
        switch (a)
        {
            case CollectValue tuple:
                return tuple.Elements.Max(TupleMax);
            case IntValue i:
                return i.Value;
            default:
                throw new NotSupportedException();
        }
    }

    public static RecursiveValue ElemScale(RecursiveValue a, RecursiveValue b)
    {
        switch (a, b)
        {
            case (CollectValue ta, CollectValue tb):
                if (ta.Count != tb.Count)
                {
                    throw new NotSupportedException();
                }

                return new CollectValue(ta.Elements.Zip(tb.Elements, ElemScale));
            case (IntValue ai, CollectValue tb):
                return new IntValue(ai.Value * Product(tb));
            case (IntValue ai, IntValue bi):
                return new IntValue(ai.Value * bi.Value);
            default:
                throw new NotSupportedException();
        }
    }

    public static RecursiveValue ShapeDiv(RecursiveValue a, RecursiveValue b)
    {
        switch (a, b)
        {
            case (CollectValue ta, CollectValue tb) when ta.Count == tb.Count:
                return new CollectValue(ta.Elements.Zip(tb.Elements, ShapeDiv));
            case (CollectValue ta, IntValue bi):
                var result = new List<RecursiveValue>();
                foreach (var v in ta.Elements)
                {
                    result.Add(ShapeDiv(v, bi));
                    bi = new IntValue(ShapeDiv(bi.Value, Product(v)));
                }

                return new CollectValue(result);

            case (IntValue ai, CollectValue tb):
                return ShapeDiv(a, new IntValue(Product(b)));
            case (IntValue ai, IntValue bi):
                var va = ai.Value;
                var vb = bi.Value;
                if (va % vb != 0 && vb % va != 0)
                {
                    throw new NotSupportedException();
                }

                return new IntValue((va + vb - 1) / vb);
            default:
                throw new NotSupportedException();
        }
    }

    public static RecursiveValue PrefixProduct(RecursiveValue a, RecursiveValue? init = null)
    {
        if (init == null)
        {
            init = 1;
        }

        switch (a, init)
        {
            case (CollectValue ta, CollectValue ti) when ta.Count == ti.Count:
                return new CollectValue(ta.Elements.Zip(ti.Elements, PrefixProduct));
            case (CollectValue ta, IntValue inti):
                var result = new List<RecursiveValue>();
                foreach (var v in ta.Elements)
                {
                    result.Add(PrefixProduct(v, inti));
                    inti = new IntValue(inti.Value * Product(v));
                }

                return new CollectValue(result);
            case (IntValue _, IntValue ti):
                return ti;
            default:
                throw new NotSupportedException();
        }
    }

    public static RecursiveValue SuffixProduct(RecursiveValue a, RecursiveValue? init = null)
    {
        if (init == null)
        {
            init = 1;
        }

        switch (a, init)
        {
            case (CollectValue ta, CollectValue ti) when ta.Count == ti.Count:
                return new CollectValue(ta.Elements.Reverse().Zip(ti.Elements.Reverse(), SuffixProduct).Reverse());
            case (CollectValue ta, IntValue inti):
                var result = new List<RecursiveValue>();
                foreach (var v in ta.Elements.Reverse())
                {
                    result.Add(SuffixProduct(v, inti));
                    inti = new IntValue(inti.Value * Product(v));
                }

                result.Reverse();
                return new CollectValue(result);
            case (IntValue _, IntValue ti):
                return ti;
            default:
                throw new NotSupportedException();
        }
    }

    public static RecursiveValue Idx2Crd(RecursiveValue idx, RecursiveValue shape, RecursiveValue? stride = null)
    {
        if (stride == null)
        {
            stride = PrefixProduct(shape);
        }

        switch (idx, shape, stride)
        {
            case (CollectValue ti, CollectValue ts, CollectValue td):
                if (ti.Count != ts.Count || ti.Count != td.Count)
                {
                    throw new NotSupportedException();
                }

                return new CollectValue(ti.Elements.Zip(ts.Elements.Zip(td.Elements, (s, d) => (shape: s, stride: d)), (i, pair) => Idx2Crd(i, pair.shape, pair.stride)));
            case (CollectValue _, IntValue _, var _):
                throw new NotSupportedException();
            case (IntValue ii, CollectValue ts2, CollectValue td2) when ts2.Count == td2.Count:
                return new CollectValue(ts2.Elements.Zip(td2.Elements, (s, d) => Idx2Crd(idx, s, d)));
            case (IntValue ii, IntValue si, IntValue sd):
                var vi = ii.Value;
                var vs = si.Value;
                var vd = sd.Value;
                return new IntValue(vi / vd % vs);
            default:
                throw new NotSupportedException();
        }
    }

    public static long Crd2Idx(RecursiveValue crd, RecursiveValue shape, RecursiveValue? stride = null)
    {
        if (stride == null)
        {
            stride = PrefixProduct(shape);
        }

        if (crd is UnderScoreValue)
        {
            crd = 0;
        }

        switch (crd, shape, stride)
        {
            case (CollectValue tc, CollectValue ts, CollectValue td) when tc.Count == ts.Count && tc.Count == td.Count:
                return tc.Elements.Zip(ts.Elements.Zip(td.Elements, (s, d) => (shape: s, stride: d)), (c, pair) => Crd2Idx(c, pair.shape, pair.stride)).Sum();
            case (IntValue ci, CollectValue ts2, CollectValue td2) when ts2.Count == td2.Count:
                var vc = ci.Value;
                long result = 0;
                for (int i = 0; i < ts2.Count - 1; i++)
                {
                    result += Crd2Idx(new IntValue(vc % Product(ts2.Elements[i])), ts2.Elements[i], td2.Elements[i]);
                    vc /= Product(ts2.Elements[i]);
                }

                return result + Crd2Idx(new IntValue(vc), ts2.Elements.Last(), td2.Elements.Last());

            case (IntValue ci, IntValue sInt, IntValue sdInt):
                var vd = sdInt.Value;
                return ci.Value * vd;
            default:
                throw new NotSupportedException();
        }
    }

    public static RecursiveValue Crd2Crd(RecursiveValue crd, RecursiveValue dstShape, RecursiveValue? srcShape = null)
    {
        switch (crd, dstShape)
        {
            case (CollectValue tc, CollectValue td) when tc.Count == td.Count:
                return new CollectValue(tc.Elements.Zip(td.Elements).Select(p => Crd2Crd(p.First, p.Second)));
            case (CollectValue _, IntValue _):
                throw new NotSupportedException();
            case (IntValue _, CollectValue td):
                return Idx2Crd(crd, dstShape);
            case (IntValue _, IntValue _):
                if (srcShape == null)
                {
                    throw new NotSupportedException();
                }

                return new IntValue(Crd2Idx(crd, srcShape));
            default:
                throw new NotSupportedException();
        }
    }

    public static CollectValue Slice(RecursiveValue crd, RecursiveValue trg)
    {
        switch (crd, trg)
        {
            case (CollectValue tc, CollectValue tt) when tc.Count == tt.Count:
                return new CollectValue(tc.Elements.Zip(tt.Elements).Select(p => Slice(p.First, p.Second)).Where(x => x.Elements.Any()).SelectMany(x => x.Elements));
            case (CollectValue _, IntValue _):
                throw new NotSupportedException();
            case (null, _):
                return new CollectValue([trg]);
            default:
                return new CollectValue([]);
        }
    }

    public static bool HasNone(RecursiveValue a)
    {
        switch (a)
        {
            case CollectValue tuple:
                return tuple.Elements.Any(HasNone);
            case IntValue i:
                return false;
            case null:
                return true;
            default:
                throw new NotSupportedException();
        }
    }

    public static Layout MakeLayout(params Layout[] layouts)
    {
        var shapes = layouts.Select(l => l.Shape).ToArray();
        var strides = layouts.Select(l => l.Stride).ToArray();
        return new Layout(new CollectValue(shapes), new CollectValue(strides));
    }

    public static long Size(Layout layout) => layout.Size();

    public static long Cosize(Layout layout) => layout.Cosize();

    public static Layout Coalesce(Layout layout, RecursiveValue? profile = null)
    {
        if (profile is CollectValue tp)
        {
            if (!(layout.Rank >= tp.Count))
            {
                throw new NotSupportedException();
            }

            var coalesced = tp.Elements
                .Select((p, i) => Coalesce(layout[i], p))
                .Concat(Enumerable.Range(tp.Count, layout.Rank - tp.Count)
                    .Select(i => layout[i]));
            return MakeLayout(coalesced.ToArray());
        }

        var resultShape = new List<long> { 1 };
        var resultStride = new List<long> { 0 };
        var flatShape = Flatten(layout.Shape);
        var flatStride = Flatten(layout.Stride);
        foreach (var (shape, stride) in flatShape.Elements.Zip(flatStride.Elements).Select(p => ((IntValue)p.First, (IntValue)p.Second)))
        {
            if (shape.Value == 1)
            {
                continue;
            }

            if (resultShape.Last() == 1)
            {
                resultShape[resultShape.Count - 1] = shape.Value;
                resultStride[resultStride.Count - 1] = stride.Value;
            }
            else if (resultShape.Last() * resultStride.Last() == stride.Value)
            {
                resultShape[resultShape.Count - 1] = resultShape.Last() * shape.Value;
            }
            else
            {
                resultShape.Add(shape.Value);
                resultStride.Add(stride.Value);
            }
        }

        if (resultShape.Count == 1)
        {
            return new Layout(new IntValue(resultShape[0]), new IntValue(resultStride[0]));
        }
        else
        {
            return new Layout(new CollectValue(resultShape.Select(s => new IntValue(s))), new CollectValue(resultStride.Select(s => new IntValue(s))));
        }
    }

    public static Layout Filter(Layout layout, RecursiveValue? profile = null)
    {
        if (profile is CollectValue tp)
        {
            Trace.Assert(layout.Rank >= tp.Count);
            var filtered = tp.Elements
                .Select((p, i) => Filter(layout[i], p))
                .Concat(Enumerable.Range(tp.Count, layout.Rank - tp.Count)
                    .Select(i => layout[i]));
            return MakeLayout(filtered.ToArray());
        }

        var resultShape = new List<long>();
        var resultStride = new List<long>();
        var flatShape = Flatten(layout.Shape);
        var flatStride = Flatten(layout.Stride);
        foreach (var (shape, stride) in flatShape.Elements.Zip(flatStride.Elements).Select(p => ((IntValue)p.First, (IntValue)p.Second)))
        {
            if (!(shape.Value == 1 || stride.Value == 0))
            {
                resultShape.Add(shape.Value);
                resultStride.Add(stride.Value);
            }
        }

        if (resultShape.Count == 0)
        {
            return new Layout(new IntValue(1), new IntValue(0));
        }
        else
        {
            return Coalesce(new Layout(new CollectValue(resultShape.Select(s => new IntValue(s))), new CollectValue(resultStride.Select(s => new IntValue(s)))));
        }
    }

    public static Layout Composition(Layout layoutA, object layoutB)
    {
        switch (layoutB)
        {
            case null:
                return layoutA;
            case IntValue iB:
                return Composition(layoutA, new Layout(iB));
            case CollectValue tB:
                Trace.Assert(layoutA.Rank >= tB.Count);
                var composed = tB.Elements
                    .Select((b, i) => Composition(layoutA[i], b))
                    .Concat(Enumerable.Range(tB.Count, layoutA.Rank - tB.Count)
                        .Select(i => layoutA[i]));
                return MakeLayout(composed.ToArray());
            case Layout lB:
                if (lB.Shape is CollectValue lBshape)
                {
                    return MakeLayout(lB.Select(bi => Composition(layoutA, bi)).ToArray());
                }

                break;
            default:
                break;
        }

        var layoutBObj = (Layout)layoutB;
        if (layoutBObj.Stride.Equals(new IntValue(0)))
        {
            return new Layout(layoutBObj.Shape, new IntValue(0));
        }
        else
        {
            var resultShape = new List<long>();
            var resultStride = new List<long>();
            var restShape = Product(layoutBObj.Shape);
            var restStride = Product(layoutBObj.Stride);
            var flatA = Coalesce(layoutA);
            var flatAShape = Flatten(flatA.Shape);
            var flatAStride = Flatten(flatA.Stride);
            for (int i = 0; i < flatAShape.Count - 1; i++)
            {
                var currShape = ((IntValue)flatAShape[i]).Value;
                var currStride = ((IntValue)flatAStride[i]).Value;
                Trace.Assert(currShape % restStride == 0 || restStride % currShape == 0);
                var newShape = Math.Min(Math.Max(1, currShape / restStride), restShape);

                if (newShape != 1)
                {
                    resultShape.Add(newShape);
                    resultStride.Add(restStride * currStride);
                }

                restShape = Utilities.MathUtility.CeilDiv(restShape, Math.Abs(newShape));
                restStride = Utilities.MathUtility.CeilDiv(Math.Abs(restStride), currShape) * Signum(restStride);
            }

            if (restShape != 1 || resultShape.Count == 0)
            {
                resultShape.Add(restShape);
                resultStride.Add(restStride * ((IntValue)flatAStride[flatAStride.Count - 1]).Value);
            }

            if (resultShape.Count == 1)
            {
                return new Layout(new IntValue(resultShape[0]), new IntValue(resultStride[0]));
            }
            else
            {
                return new Layout(new CollectValue(resultShape.Select(s => new IntValue(s))), new CollectValue(resultStride.Select(s => new IntValue(s))));
            }
        }
    }

    public static Layout Complement(Layout layout, long maxIdx = 1)
    {
        var resultShape = new List<long>();
        var resultStride = new List<long>();
        var currentIdx = 1L;

        var sortedDS = Flatten(layout.Stride).Zip(Flatten(layout.Shape)).Select(p => (st: ((IntValue)p.First).Value, sh: ((IntValue)p.Second).Value)).OrderBy(x => x.st);
        foreach (var (stride, shape) in sortedDS)
        {
            if (stride == 0 || shape == 1)
            {
                continue;
            }

            var inBound = currentIdx <= shape * stride;
            Trace.Assert(inBound);

            resultShape.Add(stride / currentIdx);
            resultStride.Add(currentIdx);
            currentIdx = shape * stride;
        }

        resultShape.Add(Utilities.MathUtility.CeilDiv(maxIdx, currentIdx));
        resultStride.Add(currentIdx);

        return Coalesce(new Layout(new CollectValue(resultShape.Select(s => new IntValue(s))), new CollectValue(resultStride.Select(s => new IntValue(s)))));
    }

    public static Layout RightInverse(Layout layout)
    {
        if (layout == null)
        {
            return null!;
        }

        if (layout.Shape is IntValue si)
        {
            return new Layout(si);
        }

        var resultShape = new List<long>();
        var resultStride = new List<long>();
        var currentIdx = 1L;

        var flatShape = Flatten(layout.Shape).OfType<IntValue>().Select(x => x.Value).ToArray();
        var flatStride = Flatten(layout.Stride).OfType<IntValue>().Select(x => x.Value).ToArray();
        var prefixProd = Flatten(PrefixProduct(layout.Shape)).OfType<IntValue>().Select(x => x.Value).ToArray();
        var sortedDSA = flatStride.Zip(flatShape).Select(p => (st: p.First, sh: p.Second)).Zip(prefixProd, (x, r) => (x.st, x.sh, r)).OrderBy(x => x.st);
        foreach (var (stride, shape, rstride) in sortedDSA)
        {
            if (shape == 1)
            {
                continue;
            }

            if (currentIdx != stride)
            {
                break;
            }

            resultShape.Add(shape);
            resultStride.Add(rstride);
            currentIdx = shape * stride;
        }

        return Coalesce(new Layout(new CollectValue(resultShape.Select(s => new IntValue(s))), new CollectValue(resultStride.Select(s => new IntValue(s)))));
    }

    public static Layout LeftInverse(Layout layout)
    {
        if (layout == null)
        {
            return null!;
        }

        if (layout.Shape is IntValue si)
        {
            return new Layout(si);
        }

        return RightInverse(MakeLayout(layout, Complement(layout)));
    }

    public static Layout LogicalDivide(Layout layoutA, RecursiveValue valueB)
    {
        switch (valueB)
        {
            case IntValue iB:
                return LogicalDivide(layoutA, new Layout(iB));
            case CollectValue tupleB:
                Trace.Assert(layoutA.Rank >= tupleB.Count);
                var divided = tupleB.Elements
                    .Select((b, i) => LogicalDivide(layoutA[i], b))
                    .Concat(Enumerable.Range(tupleB.Count, layoutA.Rank - tupleB.Count)
                        .Select(i => layoutA[i]));
                return MakeLayout(divided.ToArray());
            case Layout layoutB:
                return Composition(layoutA, MakeLayout(layoutB, Complement(layoutB, Size(layoutA))));
            default:
                throw new NotSupportedException();
        }
    }

    // public static Layout LogicalProduct(Layout layoutA, object layoutB)
    // {
    //     if (layoutB == null)
    //     {
    //         return layoutA;
    //     }
    //     if (layoutB is Int iB)
    //     {
    //         return LogicalProduct(layoutA, new Layout(iB));
    //     }
    //     if (layoutB is Tuple tB)
    //     {
    //         Trace.Assert(layoutA.Count >= tB.Count);
    //         var producted = tB.Elements
    //             .Select((b, i) => LogicalProduct(layoutA[i], b))
    //             .Concat(Enumerable.Range(tB.Count, layoutA.Count - tB.Count)
    //                 .Select(i => layoutA[i]));
    //         return MakeLayout(producted.ToArray());
    //     }
    //     return MakeLayout(layoutA, Composition(Complement(layoutA, Size(layoutA) * Cosize((Layout)layoutB)), (Layout)layoutB));
    // }
    //  public static Layout ZippedProduct(Layout layoutA, object layoutB) => HierUnzip(LogicalProduct, layoutA, layoutB);
    // public static Layout TiledProduct(Layout layoutA, object layoutB)
    // {
    //     var result = ZippedProduct(layoutA, layoutB);
    //     var layouts = new List<Layout> { result[0] };
    //     layouts.AddRange(Enumerable.Range(0, result[1].Count).Select(i => result[1][i]));
    //     return MakeLayout(layouts.ToArray());
    // }
    public static Layout HierUnzip(Func<Layout, RecursiveValue, Layout> splitter, Layout layoutA, RecursiveValue valueB)
    {
        if (valueB is UnderScoreValue)
        {
            return MakeLayout(new Layout(1, 0), layoutA);
        }

        if (valueB is CollectValue tB)
        {
            Trace.Assert(layoutA.Rank >= tB.Count);
            var split = MakeLayout(Enumerable.Range(0, tB.Count).Select(i => HierUnzip(splitter, layoutA[i], tB[i])).ToArray());
            var gathered0 = MakeLayout(Enumerable.Range(0, tB.Count).Select(i => split[i][0]).ToArray());
            var gathered1 = MakeLayout(Enumerable.Range(0, tB.Count).Select(i => split[i][1])
                .Concat(Enumerable.Range(tB.Count, layoutA.Rank - tB.Count).Select(i => layoutA[i])).ToArray());
            return MakeLayout(gathered0, gathered1);
        }

        return splitter(layoutA, valueB);
    }

    /// <summary>
    /// Tiled divide. [A,B] -> [(tileA, tileB), (timesA, timesB)].
    /// </summary>
    public static Layout ZippedDivide(Layout layoutA, RecursiveValue valueB) => HierUnzip(LogicalDivide, layoutA, valueB);

    /// <summary>
    /// Tiled divide. [A,B] -> [(tileA, tileB), timesA, timesB].
    /// </summary>
    public static Layout TiledDivide(Layout layoutA, RecursiveValue valueB)
    {
        var result = ZippedDivide(layoutA, valueB);
        var layouts = new List<Layout> { result[0] };
        layouts.AddRange(Enumerable.Range(0, result[1].Rank).Select(i => result[1][i]));
        return MakeLayout(layouts.ToArray());
    }

    public static (Layout Layout, long Offset) SliceAndOffset(RecursiveValue crd, Layout layout)
    {
        return (new Layout(Slice(crd, layout.Shape), Slice(crd, layout.Stride)), Crd2Idx(crd, layout.Shape, layout.Stride));
    }

    public static Layout GetTiler(RecursiveValue value, SBPSplit split, Placement placement)
    {
        if (value is not IntValue { Value: var dim })
        {
            throw new NotSupportedException();
        }

        var tiles = new List<IntValue>();
        foreach (var axis in split.Axes)
        {
            var outer = placement.Hierarchy[axis];
            Trace.Assert(dim % outer == 0);
            tiles.Add(new(outer));
            dim /= outer;
        }

        var shape = new CollectValue(tiles);
        return new Layout(shape, SuffixProduct(shape, dim));
    }

    public static CollectValue GetTiler(RecursiveValue shape, IRArray<SBP> axisPolicies, Placement placement)
    {
        return new CollectValue(axisPolicies.Select((sbp, i) => sbp switch
        {
            SBPSplit split when shape is CollectValue tShape => GetTiler(tShape[i], split, placement),
            _ => RecursiveValue.UnderScore,
        }));
    }

    public static RecursiveValue Minimum(RecursiveValue a, RecursiveValue b)
    {
        switch (a, b)
        {
            case (CollectValue ta, CollectValue tb) when ta.Count == tb.Count:
                return new CollectValue(Enumerable.Range(0, ta.Count).Select(i => Minimum(ta[i], tb[i])));
            case (IntValue ia, IntValue ib):
                return new IntValue(Math.Min(ia.Value, ib.Value));
            default:
                throw new NotSupportedException();
        }
    }

    public static RecursiveValue Maximum(RecursiveValue a, RecursiveValue b)
    {
        switch (a, b)
        {
            case (CollectValue ta, CollectValue tb) when ta.Count == tb.Count:
                return new CollectValue(Enumerable.Range(0, ta.Count).Select(i => Maximum(ta[i], tb[i])));
            case (IntValue ia, IntValue ib):
                return new IntValue(Math.Max(ia.Value, ib.Value));
            default:
                throw new NotSupportedException();
        }
    }

    private static long ShapeDiv(long a, long b)
    {
        if (a % b != 0 && b % a != 0)
        {
            throw new NotSupportedException();
        }

        return (a + b - 1) / b;
    }
}
