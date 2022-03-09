// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using LanguageExt.Pipes;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.Math;
using ShapeOf = Nncase.IR.Tensors.ShapeOf;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Shape GetShape(in tflite.Operator op, int index)
        {
            return GetTensorShape(GetInputTensor(op, index));
        }
        
        private Expr VisitSpaceToBatchND(in tflite.Operator op)
        {
            var (input, blockShape) = GetInputExprs(op, 0, 1);
            blockShape = Cast(blockShape, DataTypes.Int64);
            var paddings = GetInputExprs(op, 2);
            var spatialSize = GetShape(op, 1).Prod().FixedValue;
            var remainShapeSize = GetShape(op, 0).Count - spatialSize - 1;
            
            // new_paddings
            // batch
            var zeroPadding = Tensor.FromSpan<int>(new[] {0, 0}, new[] {1, 2});
            // spatial
            var spatial = Slice(paddings, new[]{0, 0}, new[] { spatialSize, 2 }, new[] { 0, 1 }, new[] { 1, 1});
            // remaining
            var remaining = Concat(
                new Tuple(
                    RangeExec(remainShapeSize, _ => zeroPadding)), 
                0);
            Expr newPaddings = ConcatList(zeroPadding, spatial, remaining);

            newPaddings = Transpose(newPaddings, new[] {1, 0});
            newPaddings = Reshape(newPaddings, Prod(ShapeOf(newPaddings)));
            
            
            Expr p = Pad(input, Cast(newPaddings, DataTypes.Int64), PadMode.Constant, 0f);
            
            // reshapped_shape
            // batch
            var batchShape1 = Util.ShapeIndex(p, 0);
            // spatial
            var spatialShape1 = ConcatList(
                RangeExec(
                    spatialSize, 
                    i =>
            {
                var blockI = Cast(Util.GetItem(blockShape, i), DataTypes.Int64);
                return ConcatList(Util.ShapeIndex(p, i + 1) / blockI, blockI);
            }));
            
            // remaining
            var remainingShape1 =
                ConcatList(
                    RangeExec(remainShapeSize, 
                        i => Util.ShapeIndex(p, 1 + (int) spatialSize + i)));
            var reshappedShape1 = ConcatList(batchShape1, spatialShape1, remainingShape1);

            // perm
            var perm = new List<int>();
            for (int i = 0; i < spatialSize; i++)
            {
                perm.Add(i * 2 + 2);
            }
            perm.Add(0);
            for (int i = 0; i < spatialSize; i++)
            {
                perm.Add(i * 2 + 1);
            }
            for (int i = 0; i < remainShapeSize; i++)
            {
                perm.Add(i + (int)spatialSize * 2 + 1);
            }
            
            // reshapped_shape2
            var batchShape2 = Util.ShapeIndex(p, 0) * ReduceSum(blockShape, 0L, 1L, false);
            var spatialShape2 = ConcatList(RangeExec(spatialSize, i => Util.ShapeIndex(p, i + 1) / Util.GetItem(blockShape, i)));
            var remainShape2 = ConcatList(RangeExec(remainShapeSize, i => Util.ShapeIndex(p, 1 + (int)spatialSize + i)));
            var reshappedShape2 = ConcatList(batchShape2, spatialShape2, remainShape2);

            var reshape = Reshape(p, reshappedShape1);
            var rt = Transpose(reshape, perm.ToArray());
            var reshape2 = Reshape(rt, reshappedShape2);
            return reshape2;
        }

        private Expr[] RangeExec(long end, Func<int, Expr> f)
        {
            return EndRange(0, (int)end).Select(f).ToArray<Expr>();
        }

        private Expr ConcatList(params Expr[] inputs)
        {
            return Concat(new Tuple(inputs), 0);
        }

        private IEnumerable<int> EndRange(int begin, int end)
        {
            return Enumerable.Range(begin, end - begin);
        }
        
        private Expr VisitBatchToSpaceND(in tflite.Operator op)
        {
            var (input0, blockShape) = GetInputExprs(op, 0, 1);
            var input2 = GetInputExprs(op, 2);

            var blockLen = GetShape(op, 1)[0].FixedValue;
            var xLen = GetShape(op, 0).Count;
            
            var oneConst = Tensor.FromSpan<long>(new [] { 1L });
            var minus1Const = Tensor.FromSpan<long>(new [] { -1L });
            var blockLenResizeConst = Tensor.FromSpan<long>(new long [] {-1, blockLen});
            var blockLenPlus1Const = Tensor.FromSpan<long>(new long [] { blockLen + 1 });
            var blockShapeConst = Cast(blockShape, DataTypes.Int64);

            var xShape = Cast(ShapeOf(input0), DataTypes.Int64);
            var spatial = Slice(xShape, oneConst, blockLenPlus1Const, 0L, 1L);
            var depth = Slice(xShape, blockLenPlus1Const, (long)xLen, 0L, 1L);
            var targetSpatial = spatial * blockShapeConst;

            var ccat1 = ConcatList(spatial, blockShapeConst);
            var re1 = Reshape(ccat1, blockLenResizeConst);
            var tr1 = Transpose(re1, new[]{1, 0});
            var interLeave = Reshape(tr1, minus1Const);
            var shape1 = ConcatList(minus1Const, interLeave, depth);

            var g1 = Range(2, 2 * blockLen + 1, 2);
            var g2 = Range(1, 2 * blockLen + 1, 2);
            var g3 = EndRange(0, xLen + blockLen).Select(x=>(long)x).ToArray()[1 + 2 * blockLen];
            var g = ConcatList(
                Cast(g1, DataTypes.Int64), 
                Tensor.FromSpan<long>(new [ ]{ 0L }), 
                Cast(g2, DataTypes.Int64), 
                Tensor.FromSpan<long>(new[] { g3 }));

            var perm = EndRange(0, xLen + blockLen).ToArray();
            perm[0] = blockLen;
            perm[1] = blockLen + 1;
            perm[2] = 0;
            foreach (var i in EndRange(3, blockLen * 2 + 1))
            {
                perm[i] = perm[i - 2] + 1;
            }

            var indices = g;
            var gather = Gather(shape1, 0, indices);
            var x2 = Reshape(input0, gather);
            var tr2 = Transpose(x2, perm);
            var shape2 = ConcatList(minus1Const, targetSpatial, depth);
            var x3 = Reshape(tr2, shape2);
            
            var crop = Cast(input2, DataTypes.Int64);
            var cropTransposed = Transpose(crop, new[] {1, 0});
            var sliceStartsConst1 = Tensor.FromSpan<long>(new long[] {0, 0});    
            var sliceStartsConst2 = ConcatList(1L, Util.ShapeIndex(cropTransposed, 1));    
            var sliceEndsConst1 = Tensor.FromSpan<long>(new long[] {1, 0});
            var sliceEndsConst2 = ConcatList(2L, Util.ShapeIndex(cropTransposed, 1));
            var axesConst = Tensor.FromSpan<long>(EndRange(1, blockLen + 1).Select(x => (long) x).ToArray());
            var strideConst = Tensor.FromSpan<long>(Enumerable.Repeat(1L, axesConst.Length).ToArray());
            var cropStarts = Slice(cropTransposed, sliceStartsConst1, sliceStartsConst2, 
                Tensor.FromSpan<long>(new long[]{0, 1}), Tensor.FromSpan<long>(new long[]{1, 1}));
            var cropEnds = Slice(cropTransposed, sliceEndsConst1, sliceEndsConst2, 
                    Tensor.FromSpan<long>(new long[]{0, 1}), Tensor.FromSpan<long>(new long[]{1, 1}));
            var cropStartsSqueeze = Squeeze(cropStarts, 0L);
            var cropEndsSqueeze = Squeeze(cropEnds, 0L);
            var endRange = targetSpatial - cropEndsSqueeze;
            var result = Slice(x3, cropStartsSqueeze, endRange, axesConst, strideConst);
            return result;
        }
    }
}