using System;
using System.Numerics.Tensors;
using System.Collections.Generic;
using System.Linq;

namespace Nncase.IR
{

    public static class TensorExtension
    {
        private static (int, int) ToIntIndex(Index Start, Index End, int dim)
        {
            int s = Start.IsFromEnd ? dim - Start.Value : Start.Value;
            int e = End.IsFromEnd ? dim - End.Value : End.Value;
            return (s, e);
        }

        private static DenseTensor<T> Get<T>(DenseTensor<T> ts, (int, int)[] ranges)
        {
            var shape = new int[ranges.Length];
            var new_ts = new DenseTensor<T>((
              from rg in ranges
              let sz = rg.Item2 - rg.Item1
              select sz).ToArray());

            int size = ranges.Aggregate(1, (sz, rg) => sz * (rg.Item2 - rg.Item1));

            var indexs = new int[ranges.Length];
            for (int i = 0; i < ranges.Length; i++)
            {
                indexs[i] = ranges[i].Item1;
            }

            int dim = ts.Strides.Length - 1;

            for (int i = 0; i < size; i++)
            {
                var idx = 0;
                for (int j = 0; j < ts.Strides.Length; j++)
                {
                    idx += ts.Strides[j] * indexs[j];
                }
                new_ts.SetValue(i, ts.GetValue(idx));
                indexs[dim]++;
                while (dim != -1 && indexs[dim] == ranges[idx].Item2)
                {
                    indexs[dim] = ranges[idx].Item1;
                    indexs[dim - 1]++;
                    dim -= 1;
                }
            }
            return new_ts;
        }

        public static DenseTensor<T> Get<T>(DenseTensor<T> ts, params Range[] range)
        {
            var ranges = new List<Range>(range);
            while (ranges.Count < ts.Dimensions.Length)
            {
                ranges.Add(..);
            }
            while (ranges.Count > ts.Dimensions.Length)
            {
                ranges.RemoveAt(ranges.Count - 1);
            }

            var normRanges = new (int, int)[ranges.Count];
            for (int i = 0; i < ranges.Count; i++)
            {
                var (s, e) = ToIntIndex(ranges[i].Start, ranges[i].End, ts.Dimensions[i]);
                normRanges[i] = (s, e);
            }
            return Get<T>(ts, normRanges);
        }
    }

    public static class ExprExtension
    {
        /// call these function when you make sure you understand
        /// that expr is what you want, although it will check
        public static T ToScalar<T>(this Expr expr)
            where T : unmanaged => expr switch
        {
            // todo:print more expr info
            Const c => c.CheckedShape.IsScalar
                ? c.ToScalar<T>()
                : throw new InvalidOperationException("Expr is not a scalar"),
            _ => throw new InvalidOperationException("Expr is not a Const"),
        };
        
        public static DenseTensor<T> ToTensor<T>(this Expr expr)
            where T : unmanaged => expr switch
        {
            // todo:print more expr info
            Const c => (!c.CheckedShape.IsScalar) && c.CheckedShape.IsFixed 
                ? c.ToTensor<T>()
                : throw new InvalidOperationException("Expr is not a fixed shape tensor"),
            _ => throw new InvalidOperationException("Expr is not a Const"),
        };
    }

}