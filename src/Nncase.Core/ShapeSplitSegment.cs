// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
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
            // todo: segs < 2??
            var preFuncParameters = preFunc.Parameters.Select(v => (Expr)v).ToArray();
            var preVarDtype = preFunc.Parameters[info.InputIndex].CheckedDataType;
            var preVarShape = preFunc.Parameters[info.InputIndex].CheckedShape.ToArray();
            preVarShape[info.DimIndex] = Dimension.Unknown;
            var dynamicVar = new Var("split_dynamic_input", new TensorType(preVarDtype, preVarShape));
            var dynamicVarShapeOf = ShapeOf(dynamicVar);
            var dynamicVarDim = Cast(dynamicVarShapeOf, DataTypes.Int32)[info.DimIndex];
            // var bodyList = info.Segments.Select(s => MakeFunByNewVar(f, info, s));
            var body = info.Segments.Reverse().Aggregate(
                // todo: fix init
                // todo: should use <=
                // todo: get item index error
                // note 这里不能用f.body, 这样就把两个图上不相关的节点连接起来了.
                // (Expr)new Call(new Function(f.Name + $"_seg_unreachable", GetItem(f.Body, 0), f.Parameters), inputs),
                // (Expr)IR.Tuple.Void,
                (Expr)IR.F.Math.Unary(UnaryOp.Abs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 12, new[] { 1, 56, 32, 24 }).Evaluate().AsTensor()),
                (sum, seg) =>
                {
                    // create new var
                    Console.WriteLine("replace var");
                    var preVar = preFunc.Parameters[info.InputIndex];
                    var fixedShape = preVar.CheckedShape.ToValueArray();
                    fixedShape[info.DimIndex] = seg;
                    // create funciton
                    Function wrapperFunc;
                    {
                        // create inner function
                        Function innerFunc;
                        {
                            var innferFixedShapeVar = new Var(preFunc.Name + $"_seg_{seg}_inner_var", new TensorType(preVar.CheckedDataType, fixedShape));
                            var mutator = new Transform.Mutators.Substitutor(e =>
                            {
                                if (object.ReferenceEquals(e, preVar))
                                {
                                    return innferFixedShapeVar;
                                }
                                return null;
                            });
                            var newBody = mutator.Visit(preFunc.Body);
                            innerFunc = new Function(preFunc.Name + $"_seg_{seg}_inner", newBody, ImmutableArray.Create(innferFixedShapeVar));
                        }
                        // create wrapper function
                        var wrapperVarShape = preVar.CheckedShape.ToArray();
                        wrapperVarShape[info.DimIndex] = Dimension.Unknown;
                        Var wrapperVar = new Var($"seg_{seg}_wrapperVar", new TensorType(preVar.CheckedDataType, wrapperVarShape));
                        Expr wrapperBody;
                        {
                            var pads = fixedShape - Cast(ShapeOf(wrapperVar), DataTypes.Int32);
                            var paddings = Transpose(Stack(new IR.Tuple(pads, new[] { 0, 0, 0, 0 }), 0), new[] { 1, 0 });
                            var input = IR.F.NN.Pad(wrapperVar, paddings, PadMode.Constant, Cast(0f, preVar.CheckedDataType));
                            wrapperBody = new Call(innerFunc, input);
                        }
                        wrapperFunc = new Function(preFunc.Name + $"_seg_{seg}", wrapperBody, ImmutableArray.Create(wrapperVar));
                    }

                    var then = new Call(wrapperFunc, preFunc.Parameters[info.InputIndex]);
                    Console.WriteLine("then infer");

                    // CompilerServices.DumpIR(then, "then",
                    // "/Users/lisa/Documents/nncase/tests_output/UnitTestCPUTarget/TestProcess/");
                    then.InferenceType();
                    Console.WriteLine("then infer end");
                    var f = new If(dynamicVarDim < seg, then, sum);
                    f.InferenceType();
                    return f;
                });

            return new Function(preFunc.Name + "_splited", body, ImmutableArray.Create(dynamicVar));
        }

        // public Var MakeFunByNewVar(Function f, SegmentInfo info, int segment)
        // {
        //     Console.WriteLine("replace var");
        //     var oldVar = f.Parameters[info.InputIndex];
        //     var newShape = oldVar.CheckedShape.ToArray();
        //     newShape[info.DimIndex] = segment;

        //     // todo 这里目前就一个var 没问题, 多个var会有问题
        //     var newVar = new Var("split_" + oldVar.Name, new TensorType(oldVar.CheckedDataType, newShape));
        //     CompilerServices.InferenceType(newVar);
        //     // var newParams = f.Parameters.ToArray();
        //     // newParams[info.InputIndex] = newVar;
        //     return newVar;
        // }
    }
}
