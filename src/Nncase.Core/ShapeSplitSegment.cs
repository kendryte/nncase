// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics;
using Nncase.IR;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;

namespace Nncase
{
    public record SegmentInfo(int InputIndex, int DimIndex, int[] Segments);

    public class ShapeSplitSegment
    {
        public Function Run(Function preFunc, SegmentInfo info)
        {
            Debug.Assert(info.Segments.Length >= 2);
            var inShape = ShapeOf(preFunc.Parameters[info.InputIndex]);
            var preVar = preFunc.Parameters[info.InputIndex];

            var dim = Cast(inShape, DataTypes.Int32)[info.DimIndex];
            var body = info.Segments.Reverse().Aggregate(
                // todo: this will be fold
                (Expr)IR.F.Math.Require(true, 0, "input dim large than limit"),
                (sum, seg) =>
                {
                    var inputs = preFunc.Parameters.Select(v => (Expr)v).ToArray();
                    int[] fixedShape = ComputeFixedShape(preFunc, info, seg);
                    var innerFunc = SplitFuncImpl(preFunc, seg, fixedShape, preVar);
                    var fitFixedShape = FitFixedShape(preFunc, info, seg, fixedShape, innerFunc);
                    var then = new Call(fitFixedShape, inputs);
                    return new If(dim <= seg, then, sum);
                });

            return new Function(body, preFunc.Parameters);
        }

        private static int[] ComputeFixedShape(Function preFunc, SegmentInfo info, int seg)
        {
            var fixedShape = preFunc.Parameters[info.InputIndex].CheckedShape.ToValueArray();
            fixedShape[info.DimIndex] = seg;
            return fixedShape;
        }

        private static Function FitFixedShape(Function preFunc, SegmentInfo info, int seg, int[] fixedShape, Function innerFunc)
        {
            // rename
            var wrapParams = preFunc.Parameters.ToArray();
            wrapParams[info.InputIndex] = wrapParams[info.InputIndex] with { Name = "split_dynamic_input" };
            var targetInput = wrapParams[info.InputIndex];

            // compute paddings;
            var pads = fixedShape - Cast(ShapeOf(targetInput), DataTypes.Int32);
            var paddings = Transpose(Stack(new IR.Tuple(pads, new[] { 0, 0, 0, 0 }), 0), new[] { 1, 0 });
            var fixedInput = IR.F.NN.Pad(targetInput, paddings, PadMode.Constant, Cast(0f, targetInput.CheckedDataType));
            var wrapperBody = new Call(innerFunc, fixedInput);


            // forward origin input.
            var wrapperFunc = new Function(preFunc.Name + $"_seg_{seg}", wrapperBody, wrapParams);
            return wrapperFunc;
        }

        private Function SplitFuncImpl(Function preFunc, int seg, int[] fixedShape, Var preVar)
        {

            var innerFixedShapeVar = new Var(preFunc.Name + $"_seg_{seg}_inner_var",
                new TensorType(preVar.CheckedDataType, fixedShape));
            var newBody = ReplaceExpr(preFunc.Body, preVar, innerFixedShapeVar);

            // replace Fix var.
            var innerFunc = new Function(preFunc.Name + $"_seg_{seg}_inner", newBody,
                ImmutableArray.Create(innerFixedShapeVar));
            return innerFunc;
        }

        private Expr ReplaceExpr(Expr body, Expr target, Expr expr)
        {
            var mutator = new Transform.Mutators.Substitutor(e =>
            {
                if (ReferenceEquals(e, target))
                {
                    return expr;
                }

                return null;
            });
            return mutator.Visit(body);
        }
    }
}
