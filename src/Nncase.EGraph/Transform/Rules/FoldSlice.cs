using System.Linq;
using Nncase.IR;
using Nncase.Transform.Pattern.Tensors;
using static Nncase.Transform.Pattern.Utility;
using System.Numerics.Tensors;
using static Nncase.IR.F.Tensors;

namespace Nncase.Transform.Rule
{

    public class FoldSliceSlice : EGraphRule
    {
        SliceWrapper slice1, slice2;
        public FoldSliceSlice()
        {
            slice1 = IsSlice(IsWildCard());
            slice2 = IsSlice(slice1);
            Pattern = slice2;
        }

        private bool IsNoSlice(SliceWrapper slice, int dim)
        {

            var inshape = slice.Input().CheckedShape;

            var begin = slice.Begins<Const>().ToTensor<int>();
            var end = slice.Ends<Const>().ToTensor<int>();
            var axes = slice.Axes<Const>().ToTensor<int>();
            var strides = slice.Strides<Const>().ToTensor<int>();
            return Enumerable.Range(0, (int)begin.Length).All(i => begin[i] == 0 && end[i] == inshape[i].FixedValue && strides[i] == 1);
        }

        private bool CanMerge(SliceWrapper slice1, SliceWrapper slice2)
        {
            var inshape = slice1.Input().CheckedShape;
            var axes1 = slice1.Axes<Const>().ToTensor<int>();
            var axes2 = slice2.Axes<Const>().ToTensor<int>();
            return Enumerable.Range(0, inshape.Rank).All(
              dim => (IsNoSlice(slice1, dim) || IsNoSlice(slice2, dim)) && (axes1[dim] == axes2[dim])
            );
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            slice1.Bind(result);
            slice2.Bind(result);
            if (!CanMerge(slice1, slice2))
                return null;

            var old_begin1 = slice1.Begins<Const>().ToTensor<int>();
            var old_end1 = slice1.Ends<Const>().ToTensor<int>();
            var old_strides1 = slice1.Strides<Const>().ToTensor<int>();

            var old_begin2 = slice2.Begins<Const>().ToTensor<int>();
            var old_end2 = slice2.Ends<Const>().ToTensor<int>();
            var old_strides2 = slice2.Strides<Const>().ToTensor<int>();

            var new_begin = new DenseTensor<int>(old_begin1.Dimensions);
            var new_end = new DenseTensor<int>(old_begin1.Dimensions);
            var new_strides = new DenseTensor<int>(old_begin1.Dimensions);
            for (int dim = 0; dim < slice1.Input().Rank; dim++)
            {
                new_begin[dim] = IsNoSlice(slice1, dim) ? old_begin2[dim] : old_begin1[dim];
                new_end[dim] = IsNoSlice(slice1, dim) ? old_end2[dim] : old_end1[dim];
                new_strides[dim] = IsNoSlice(slice1, dim) ? old_strides2[dim] : old_strides1[dim];
            }
            return Slice(slice1.Input(), Const.FromTensor<int>(new_begin), Const.FromTensor<int>(new_end),
                                        slice1.Axes(), Const.FromTensor<int>(new_strides));
        }
    }
}