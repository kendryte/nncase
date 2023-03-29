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

    public class ShapeBucket
    {
        private static int Count = 0;

        private Expr Default => new Call(new Function(IR.F.Math.Require(false, 0, "input dim large than limit")));

        record SingleSegment(int InputIndex, int DimIndex, int SegmentLimit);

        private Expr SplitImpl(Function preFunc, SegmentInfo[] infos, Var[] vars)
        {
            return SplitImpl(preFunc, infos, 0, 1, vars);
        }

        private Expr SplitImpl(Function preFunc, SegmentInfo[] infos, int current, int limit, Var[] outVars)
        {
            var info = infos[current];
            Check(preFunc, info);
            var body = info.Segments.OrderByDescending(x => x).Aggregate(
                Default,
                (sum, seg) =>
                {
                    var currVars = outVars.Select(v => new Var($"hash_{v.GetHashCode()}", v.TypeAnnotation)).ToArray();

                    var cond = info.Segments.Aggregate((Expr)true, (cond, _) =>
                    {
                        var dim = MakeGetDimCall(currVars, info);
                        return IR.F.Math.LogicalAnd(cond, dim <= seg);
                    });

                    // get all input fixed shape info
                    var newSegments = infos.Select(info => new SingleSegment(info.InputIndex, info.DimIndex, seg)).ToImmutableArray();

                    var thenCall = current + 1 < limit
                        ? SplitImpl(preFunc, infos, current + 1, limit, currVars)
                        : MakeSplitEntry(preFunc, currVars, newSegments);

                    var elseF = (Function)(((Call)sum).Target);
                    var elseIfVar = currVars.Select(v => new Var($"hash_else_{v.GetHashCode()}", v.TypeAnnotation)).ToArray();
                    var elseBody = ReplaceVarInFunction(elseF, elseF.Parameters.ToArray().Zip(elseIfVar).ToArray(), out var elseNewParams);
                    var elseFn = new Function(elseF.Name, elseBody, elseNewParams);
                    var elseCall = new Call(elseFn, currVars);
                    var body = new If(
                        cond,
                        thenCall,
                        elseCall);
                    var f = new Function(body, currVars);
                    return new Call(f, outVars);
                });
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

        public IRModule Run(Function preFn, SegmentInfo[] infos)
        {
            var vars = preFn.Parameters.ToArray().Select((x, i) => new Var($"main_{i}", x.TypeAnnotation)).ToArray();
            var body = SplitImpl(preFn, infos, vars);
            var newFn = new Function("main", body, vars);
            return CollectFunctionToNewModule(newFn);
        }

        private Expr MakeSplitEntry(Function preFunc, Var[] outerInput, ImmutableArray<SingleSegment> infos)
        {
            // preprocess
            var inputs = outerInput.Select((v, i) => new Var($"SplitEntry_{i}", v.TypeAnnotation)).ToArray();
            var fixedShapeList = infos.Select(info =>
                ComputeFixedShape(inputs[info.InputIndex], info.DimIndex, info.SegmentLimit)).ToArray();

            var wrapParams = inputs.ToArray();
            for (int i = 0; i < infos.Length; i++)
            {
                wrapParams[infos[i].InputIndex] = new Var($"split_dynamic_input_{i}",
                    wrapParams[infos[i].InputIndex].TypeAnnotation);
            }

            var fitFixedShape = infos.Zip(fixedShapeList)
                .Select(info => FitFixedShape(wrapParams[info.First.InputIndex], info.Second, info.First.InputIndex))
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

            // postprocess
            var impl = new Function(new Call(innerFunc, newInputs), inputs);
            var then = new Call(impl, outerInput);

            // input0: (1, len)
            // ShapeOf(input0)[1] == len
            return PostProcess(then, outerInput);
        }

        public Expr PostProcess(Expr output, Var[] outerInput)
        {
            // find first false
            var actualLen = ShapeOf(outerInput[0])[1];
            var len = Cast(actualLen, DataTypes.Int32);
            // todo: replce with expr, not constant
            // todo: out_shape_list in first should be remove
            var encOut = Slice(output[0], new[] { 0, 0, 0 }, Stack(new Tuple(1, len, 224), 0), 3);
            var dur = Slice(output[1], new[] { 0, 0 }, Stack(new Tuple(ShapeOf(output[1])[0], len), 0), 2);
            return new Tuple(encOut, dur);
        }

        private static void Check(Function f, SegmentInfo info)
        {
            Debug.Assert(info.Segments.Length >= 2, "Segments.Length >= 2");
            Debug.Assert(f.Parameters.Length >= info.InputIndex, "f.Parameters.Count <= info.InputIndex");
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

        private static Expr FitFixedShape(Expr targetInput, int[] fixedShape, int inputIndex)
        {
            targetInput.InferenceType();
            // rename
            // compute paddings;
            var pads = fixedShape - Cast(ShapeOf(targetInput), DataTypes.Int32);
            var paddings = Transpose(Stack(new IR.Tuple(Enumerable.Repeat(0, fixedShape.Length).ToArray(), pads), 0),
                new[] { 1, 0 });
            var padValue = inputIndex == 0 ? 0 : 1;
            var fixedInput = IR.F.NN.Pad(targetInput, paddings, PadMode.Constant,
                Cast(padValue, targetInput.CheckedDataType));
            return fixedInput;
        }

        private Function InnerFnImpl(Function preFunc, int[][] fixedShapeList, ImmutableArray<SingleSegment> segments)
        {
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
                    preFunc.Parameters.ToArray(),
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
