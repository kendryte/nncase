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
            return Enumerable.Range(0, (int)end).Select(f).ToArray<Expr>();
        }

        private Expr ConcatList(params Expr[] inputs)
        {
            return Concat(new Tuple(inputs), 0);
        }

        private Expr ZeroPad(Expr input, Expr pads)
        {
            return Pad(input, pads, PadMode.Constant, 0L);
        }
        
        private Expr VisitBatchToSpaceND(in tflite.Operator op)
        {
            return 1;
        }
    }
}