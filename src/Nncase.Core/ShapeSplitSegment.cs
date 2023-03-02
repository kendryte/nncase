// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;
using Tuple = Nncase.IR.Tuple;

namespace Nncase
{
    public record SegmentInfo(int InputIndex, int DimIndex, int[] Segments);

    public class ShapeSplitSegment
    {
        private static int Count = 0;

        public IRModule Run(Function preFunc, SegmentInfo info)
        {
            return null;
        }

        private Expr Default => new Call(new Function(IR.F.Math.Require(false, 0, "input dim large than limit")));

        record SingleSegment(int InputIndex, int DimIndex, int SegmentLimit)
        {
        }

        private Expr SplitImpl(Function preFunc, SegmentInfo[] infos, Var[] vars)
        {
            var segments = Array.Empty<SingleSegment>().ToImmutableArray();

            var dims = new Expr[infos.Length];
            var dimMap = new Dictionary<(int, int), int>();
            for (int i = 0; i < infos.Length; i++)
            {
                var info = infos[i];
                var inShape = ShapeOf(preFunc.Parameters[info.InputIndex]);
                var tmpDim = Cast(inShape, DataTypes.Int32)[info.DimIndex];
                dims[i] = tmpDim;
                dimMap[(info.InputIndex, info.DimIndex)] = i;
            }

            var dimTuple = new Tuple(dims);
            var dimGetItemMap = dimMap.ToDictionary(x => x.Key, x => MakeGetDim(dimTuple, dimMap[x.Key]));
            return SplitImpl(preFunc, infos, 0, infos.Length, segments, vars);
        }

        private Expr MakeGetDim(Expr dimTuple, int index)
        {
            // getItem(Tuple)
            var f = new Function(dimTuple[index], new Var("Tuples"));
            return new Call(f, dimTuple);
        }

        private Expr SplitImpl(Function preFunc, SegmentInfo[] infos, int current, int limit,
            ImmutableArray<SingleSegment> segments, Var[] outVars)
        {
            var info = infos[current];
            Check(preFunc, info);
            var body = info.Segments.OrderByDescending(x => x).Aggregate(
                Default,
                (sum, seg) =>
                {
                    // var currVars = outVars.Select(v => v with { Name = $"now_{v.Name}_hash_{v.GetHashCode()}" }).ToArray();
                    var currVars = outVars.Select(v => new Var($"hash_{v.GetHashCode()}", v.TypeAnnotation)).ToArray();
                    var dim = MakeGetDimCall(currVars, info);

                    var single = new SingleSegment(info.InputIndex, info.DimIndex, seg);
                    var newSegments = segments.Append(single).ToImmutableArray();

                    var thenCall = current + 1 < limit
                        ? SplitImpl(preFunc, infos, current + 1, limit, newSegments, currVars)
                        : MakeSplitEntry(preFunc, currVars, newSegments);



                    // CompilerServices.DumpIR(sum, $"sum_{Count++}", "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestMultiInputShapeSplit/sum");
                    // CompilerServices.DumpIR(elseF, "elseF", "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestMultiInputShapeSplit/sum");

                    var elseF = (Function)(((Call)sum).Target);
                    var elseIfVar = currVars.Select(v => new Var($"hash_else_{v.GetHashCode()}", v.TypeAnnotation)).ToArray();
                    var elseBody = ReplaceVarInFunction(elseF, elseF.Parameters.Zip(elseIfVar).ToArray(), out var elseNewParams);
                    var elseFn = new Function(elseF.Name, elseBody, elseNewParams);
                    var elseCall = new Call(elseFn, currVars);
                    // CompilerServices.DumpIR(elseFn, $"else_{Count++}", "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestMultiInputShapeSplit/sum");
                    // var elseCall = ((Call)sum) with { Parameters = currVars };
                    var body = new If(
                        dim <= seg,
                        thenCall,
                        elseCall);
                    var f = new Function(body, currVars);
                    return new Call(f, outVars);
                    // return body;
                });
            // var f = new Function(body, preFunc.Parameters);
            // var c = new Call(f, preFunc.Parameters.ToArray());
            return body;
        }

        private static Expr MakeGetDimCall(Var[] inputs, SegmentInfo info)
        {
            var input = new Var("input", inputs[info.InputIndex].TypeAnnotation);
            var inShape = ShapeOf(input);
            var dim = Cast(inShape, DataTypes.Int32)[info.DimIndex];
            var f = new Function($"GetDim_{Count++}_{info.InputIndex}_{info.DimIndex}", dim, new[] { input });
            return new Call(f, inputs[info.InputIndex]);
        }

        public IRModule SplitInfos(Function preFn, SegmentInfo[] infos)
        {
            var vars = preFn.Parameters.Select((x, i) => new Var($"main_{i}", x.TypeAnnotation)).ToArray();
            var body = SplitImpl(preFn, infos, vars);
            // body = ReplaceUtility.ReplaceExpr(body, preFn.Parameters[0], v0);
            // body = ReplaceUtility.ReplaceExpr(body, preFn.Parameters[1], v1);

            var newFn = new Function("main", body, vars);

            return CollectFunctionToNewModule(newFn);
        }

        private Expr MakeSplitEntry(Function preFunc, Var[] outerInput, ImmutableArray<SingleSegment> infos)
        {
            var inputs = outerInput.Select(v => new Var("SplitEntry", v.TypeAnnotation)).ToArray();
            var fixedShapeList = infos.Select(info =>
                ComputeFixedShape(inputs[info.InputIndex], info.DimIndex, info.SegmentLimit)).ToArray();

            var wrapParams = inputs.ToArray();
            for (int i = 0; i < infos.Length; i++)
            {
                wrapParams[infos[i].InputIndex] = new Var($"split_dynamic_input_{i}", wrapParams[infos[i].InputIndex].TypeAnnotation);
            }

            var fitFixedShape = infos.Zip(fixedShapeList)
                .Select(info => FitFixedShape(wrapParams[info.First.InputIndex], info.Second))
                .ToArray();

            var wrapperFn = new Function(preFunc.Name + $"_{MakeSegmentsStr(infos)}_wrapper",
                new IR.Tuple(fitFixedShape), wrapParams);
            var fitFixedShapeInputs = new Call(wrapperFn, inputs.ToArray());

            var newInputs = inputs.Select(v => (Expr)v).ToArray();
            for (int i = 0; i < infos.Length; i++)
            {
                newInputs[infos[i].InputIndex] = fitFixedShapeInputs[i];
            }

            var innerFunc = InnerFnImpl(preFunc, fixedShapeList, infos);
            var impl = new Function(new Call(innerFunc, newInputs), inputs);
            var then = new Call(impl, outerInput);
            return then;
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

        private static int[] ComputeFixedShape(Expr input, int dimIndex, int seg)
        {
            var originShape = input.CheckedShape;
            var dims = originShape.Select(x => x).ToArray();

            // only dims[DimIndex] is unknown
            dims[dimIndex] = seg;
            return dims.Select(x => x.FixedValue).ToArray();
        }

        private static Expr FitFixedShape(Expr targetInput, int[] fixedShape)
        {
            // rename
            // compute paddings;
            var pads = fixedShape - Cast(ShapeOf(targetInput), DataTypes.Int32);
            var paddings = Transpose(Stack(new IR.Tuple(pads, new[] { 0, 0, 0, 0 }), 0), new[] { 1, 0 });
            var fixedInput = IR.F.NN.Pad(targetInput, paddings, PadMode.Constant,
                Cast(0f, targetInput.CheckedDataType));
            return fixedInput;
            // var wrapperBody = new Call(innerFunc, fixedInput);
            //
            // // forward origin inputs.
            // var wrapperFunc = new Function(preFunc.Name + $"_seg_{seg}", wrapperBody, wrapParams);
            // return new Call(wrapperFunc, preFunc.Parameters.ToArray());
            // return wrapperFunc;
        }

        private Function InnerFnImpl(Function preFunc, int[][] fixedShapeList, ImmutableArray<SingleSegment> segments)
        {
            // todo: maybe a bug
            var segStr = MakeSegmentsStr(segments);
            int index = 0;
            var inputIndex = segments.Select(s => s.InputIndex);
            var innerFixedShapeVarList = fixedShapeList
                .Zip(inputIndex)
                .Select(indexAndShape =>
                {
                    var (fixedShape, inIndex) = indexAndShape;
                    var oldVar = preFunc.Parameters[inIndex];
                    return (oldVar, new Var(
                        preFunc.Name + $"_seg_{MakeSegmentsStr(segments)}_inner_var_{index++}",
                        new TensorType(oldVar.CheckedDataType, fixedShape)));
                })
                .ToArray();

            var newBody = ReplaceVarInFunction(preFunc, innerFixedShapeVarList, out var newParams);

            // replace Fix var.
            var innerFunc = new Function(
                preFunc.Name + $"_seg_{segStr}_inner",
                newBody,
                newParams);
            return innerFunc;
        }

        private static Expr ReplaceVarInFunction(Function preFunc, (Var oldVar, Var)[] innerFixedShapeVarList,
            out Var[] newParams)
        {
            var newBody = innerFixedShapeVarList.Aggregate(
                preFunc.Body,
                (tmpBody, vars) =>
                {
                    var (preVar, innerFixedShapeVar) = vars;
                    return ReplaceUtility.ReplaceExpr(tmpBody, preVar, innerFixedShapeVar);
                });

            newParams = ReplaceUtility.ReplaceItems(
                    preFunc.Parameters,
                    innerFixedShapeVarList
                        .Select(x => ((Expr)x.Item1, (Expr)x.Item2))
                        .ToArray())
                .Select(x => (Var)x)
                .ToArray();
            return newBody;
        }

        private static string MakeSegmentsStr(ImmutableArray<SingleSegment> segments) =>
            string.Join("_", segments.Select(x => x.SegmentLimit));
    }
}
