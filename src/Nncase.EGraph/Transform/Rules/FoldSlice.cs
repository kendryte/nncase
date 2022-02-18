// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Numerics.Tensors;
using System.Linq;
using static Nncase.Pattern.Utility;
using static Nncase.IR.F.Tensors;
using Nncase.Pattern.Tensors;
using Nncase.Pattern;
using Nncase.IR;

namespace Nncase.Transform.Rule
{
    public class FoldSliceSlice : IRewriteRule
    {
        SliceWrapper slice1, slice2;
        public FoldSliceSlice()
        {
            slice1 = IsSlice(IsWildcard());
            slice2 = IsSlice(slice1);
            Pattern = slice2;
        }

        private bool IsNoSlice(SliceWrapper slice, int dim)
        {
            var inshape = slice.Input().CheckedShape;

            var begin = slice.Begins<TensorConst>().Value.Cast<int>();
            var end = slice.Ends<TensorConst>().Value.Cast<int>();
            var axes = slice.Axes<TensorConst>().Value.Cast<int>();
            var strides = slice.Strides<TensorConst>().Value.Cast<int>();
            return Enumerable.Range(0, (int)begin.Length).All(i => begin[i] == 0 && end[i] == inshape[i].FixedValue && strides[i] == 1);
        }

        private bool CanMerge(SliceWrapper slice1, SliceWrapper slice2)
        {
            var inshape = slice1.Input().CheckedShape;
            var axes1 = slice1.Axes<TensorConst>().Value.Cast<int>();
            var axes2 = slice2.Axes<TensorConst>().Value.Cast<int>();
            return Enumerable.Range(0, inshape.Rank).All(
              dim => (IsNoSlice(slice1, dim) || IsNoSlice(slice2, dim)) && (axes1[dim] == axes2[dim])
            );
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            slice1.Bind(result);
            slice2.Bind(result);
            if (!CanMerge(slice1, slice2))
            {
                return null;
            }

            var old_begin1 = slice1.Begins<TensorConst>().Value.Cast<int>();
            var old_end1 = slice1.Ends<TensorConst>().Value.Cast<int>();
            var old_strides1 = slice1.Strides<TensorConst>().Value.Cast<int>();

            var old_begin2 = slice2.Begins<TensorConst>().Value.Cast<int>();
            var old_end2 = slice2.Ends<TensorConst>().Value.Cast<int>();
            var old_strides2 = slice2.Strides<TensorConst>().Value.Cast<int>();

            var new_begin = new Tensor<int>(old_begin1.Dimensions);
            var new_end = new Tensor<int>(old_begin1.Dimensions);
            var new_strides = new Tensor<int>(old_begin1.Dimensions);
            var rank = slice1.Input().CheckedShape.Rank;
            for (int dim = 0; dim < rank; dim++)
            {
                new_begin[dim] = IsNoSlice(slice1, dim) ? old_begin2[dim] : old_begin1[dim];
                new_end[dim] = IsNoSlice(slice1, dim) ? old_end2[dim] : old_end1[dim];
                new_strides[dim] = IsNoSlice(slice1, dim) ? old_strides2[dim] : old_strides1[dim];
            }

            return Slice(slice1.Input(), Const.FromTensor(new_begin), Const.FromTensor(new_end),
                                        slice1.Axes(), Const.FromTensor(new_strides));
        }
    }
}