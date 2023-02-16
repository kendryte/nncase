// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;

namespace Nncase
{
    public record SegmentInfo(int InputIndex, int DimIndex, int[] Segments);

    public class ShapeSplitSegment
    {
        public Function Run(Function f, SegmentInfo info)
        {
            // todo: segs < 2??
            var inputs = f.Parameters.Select(v => (Expr)v).ToArray();
            var inShape = ShapeOf(f.Parameters[info.InputIndex]);

            var dim = Cast(inShape, DataTypes.Int32)[info.DimIndex];
            var bodyList = info.Segments.Select(s => MakeFunByNewVar(f, info, s));
            var body = bodyList.Zip(info.Segments).Reverse().Aggregate(

                // todo: fix init
                // todo: should use <=
                // todo: get item index error
                (Expr)new Call(f with { Body = GetItem(f.Body, 0) }, inputs),
                (sum, now) =>
                {
                    var (fn, seg) = now;
                    var fixedShape = fn.Parameters[info.InputIndex].CheckedShape;
                    var pads = fixedShape - Cast(inShape, DataTypes.Int32);

                    // todo: fix 4d padding
                    var paddings = Transpose(Stack(new IR.Tuple(pads, new[] { 0, 0, 0, 0 }), 0), new[] { 1, 0 });
                    var input = IR.F.NN.Pad(inputs[info.InputIndex], paddings, PadMode.Constant,
                        Cast(0f, inputs[info.InputIndex].CheckedDataType));
                    var fixedInputs = inputs.ToArray();
                    fixedInputs[info.InputIndex] = input;
                    var then = new Call(fn, fixedInputs);
                    Console.WriteLine("then infer");

                    // CompilerServices.DumpIR(then, "then",
                    // "/Users/lisa/Documents/nncase/tests_output/UnitTestCPUTarget/TestProcess/");
                    then.InferenceType();
                    Console.WriteLine("then infer end");
                    var f = new If(dim < seg, then, sum);
                    f.InferenceType();
                    return f;
                });

            return new Function(body, f.Parameters);
        }

        public Function MakeFunByNewVar(Function f, SegmentInfo info, int segment)
        {
            Console.WriteLine("replace var");
            var oldVar = f.Parameters[info.InputIndex];
            var newShape = oldVar.CheckedShape.ToArray();
            newShape[info.DimIndex] = segment;

            var newVar = new Var("split_" + oldVar.Name, new TensorType(oldVar.CheckedDataType, newShape));
            var newBody = ReplaceUtility.ReplaceTarget(f.Body, oldVar, newVar);

            var newParams = f.Parameters.ToArray();
            newParams[info.InputIndex] = newVar;

            // newParams = newParams.Append(newVar).ToArray();
            var rf = new Function(f.Name + $"_seg_{segment}", newBody, newParams);
            rf.InferenceType();
            return rf;
        }
    }
}
