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
        public IRModule Run(Function preFunc, SegmentInfo info)
        {
            Check(preFunc, info);
            var inShape = ShapeOf(preFunc.Parameters[info.InputIndex]);
            var preVar = preFunc.Parameters[info.InputIndex];

            var dim = Cast(inShape, DataTypes.Int32)[info.DimIndex];
            var body = info.Segments.Reverse().Aggregate(
                (Expr)IR.F.Math.Require(false, 0, "input dim large than limit"),
                (sum, seg) =>
                {
                    var inputs = preFunc.Parameters.Select(v => (Expr)v).ToArray();
                    int[] fixedShape = ComputeFixedShape(preFunc, info, seg);
                    var innerFunc = SplitFuncImpl(preFunc, seg, fixedShape, preVar);
                    var fitFixedShape = FitFixedShape(preFunc, info, seg, fixedShape, innerFunc);
                    var then = new Call(fitFixedShape, inputs);
                    return new If(dim <= seg, then, sum);
                });

            var newFn = new Function(body, preFunc.Parameters);
            return CollectFunctionToNewModule(newFn);
        }

        private static void Check(Function f, SegmentInfo info)
        {
            Debug.Assert(info.Segments.Length >= 2, "Segments.Length >= 2");
            Debug.Assert(f.Parameters.Count >= info.InputIndex, "f.Parameters.Count <= info.InputIndex");
        }

        private static IRModule CollectFunctionToNewModule(Function splitMain)
        {
            var c = new FunctionCollector();
            c.Visit(splitMain);
            var module = new IRModule();
            foreach (var fn in c.Functions)
            {
                module.Add(fn);
            }

            module.Entry = splitMain;
            return module;
        }

        private static int[] ComputeFixedShape(Function preFunc, SegmentInfo info, int seg)
        {
            var originShape = preFunc.Parameters[info.InputIndex].CheckedShape;
            var dims = originShape.Select(x => x).ToArray();

            // only dims[DimIndex] is unknown
            dims[info.DimIndex] = seg;
            return dims.Select(x => x.FixedValue).ToArray();
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

            // forward origin inputs.
            var wrapperFunc = new Function(preFunc.Name + $"_seg_{seg}", wrapperBody, wrapParams);
            return wrapperFunc;
        }

        private Function SplitFuncImpl(Function preFunc, int seg, int[] fixedShape, Var preVar)
        {
            var innerFixedShapeVar = new Var(
                preFunc.Name + $"_seg_{seg}_inner_var",
                new TensorType(preVar.CheckedDataType, fixedShape));
            var newBody = ReplaceUtility.ReplaceExpr(preFunc.Body, preVar, innerFixedShapeVar);

            // replace Fix var.
            var innerFunc = new Function(
                preFunc.Name + $"_seg_{seg}_inner",
                newBody,
                ImmutableArray.Create(innerFixedShapeVar));
            return innerFunc;
        }
    }
}
