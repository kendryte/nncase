// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;
using System.Linq.Expressions;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;
using Tuple = Nncase.IR.Tuple;

namespace Nncase
{
    public record SegmentInfo(int InputIndex, int DimIndex, int[] Segments);

    public class ShapeBucket
    {
        private static int Count = 0;

        private Expr Default => IR.F.Math.Require(false, 0, "input dim large than limit");

        record SingleSegment(int InputIndex, int DimIndex, int SegmentLimit);

        private Expr SplitImpl(Function preFunc, SegmentInfo[] infos, Var[] vars)
        {
            return SplitImpl(preFunc, infos, 0, 1, vars);
        }

        private Expr SplitImpl(Function originPreFunc, SegmentInfo[] infos, int current, int limit, Var[] outVars)
        {
            var info = infos[current];
            Check(originPreFunc, info);
            var body = info.Segments.OrderByDescending(x => x).Aggregate(
                Default,
                (sum, seg) =>
                {
                    var currVars = outVars;

                    var cond = info.Segments.Aggregate((Expr)true, (cond, _) =>
                    {
                        var dim = MakeGetDimCall(currVars, info);
                        return IR.F.Math.LogicalAnd(cond, dim <= seg);
                    });

                    // get all input fixed shape info
                    var newSegments = infos.Select(info => new SingleSegment(info.InputIndex, info.DimIndex, seg)).ToImmutableArray();

                    var preFunc = originPreFunc.Clone();
                    CompilerServices.DumpIR(preFunc, MakeSegmentsStr(newSegments), "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestSplitEncWithMask/dump");
                    CompilerServices.DumpIR(preFunc.Clone(), MakeSegmentsStr(newSegments) + "Clone", "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestSplitEncWithMask/dump");

                    var thenBody = current + 1 < limit
                        ? SplitImpl(preFunc, infos, current + 1, limit, currVars)
                        : MakeSplitEntry(preFunc, currVars, newSegments);

                    var elseBody = sum;
                    return new If(cond, thenBody, elseBody);
                });
            return body;
        }

        private static Expr MakeGetDimCall(Var[] inputs, SegmentInfo info)
        {
            var input = inputs[info.InputIndex];
            var inShape = ShapeOf(input);
            var dim = Cast(inShape, DataTypes.Int32)[info.DimIndex];
            return dim;
        }

        private CompileOptions _options;
        public IRModule Run(Function preFn, SegmentInfo[] infos, CompileOptions options)
        {
            _options = options;
            var vars = preFn.Parameters.ToArray();
            var body = SplitImpl(preFn, infos, vars);
            var newFn = new Function("main", body, vars);
            return CollectFunctionToNewModule(newFn);
        }

        private (Expr[], int[][]) Preprocess(Function preFunc, Expr[] outerInput, ImmutableArray<SingleSegment> infos)
        {
            var inputs = outerInput;
            var fixedShapeList = infos.Select(info =>
                ComputeFixedShape(inputs[info.InputIndex], info.DimIndex, info.SegmentLimit)).ToArray();

            var fitFixedShape = infos.Zip(fixedShapeList)
                .Select(info => FitFixedShape(outerInput[info.First.InputIndex], info.Second, info.First.InputIndex))
                .ToArray();

            Expr fitFixedShapeInputs = new IR.Tuple(fitFixedShape);

            var newInputs = inputs.Select(v => (Expr)v).ToArray();
            for (int i = 0; i < infos.Length; i++)
            {
                newInputs[infos[i].InputIndex] = fitFixedShapeInputs[i];
            }
            return (newInputs, fixedShapeList);
        }

        internal class DatasetProvider :ICalibrationDatasetProvider
        {
            public int? Count { get; }

            public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }

            public DatasetProvider(IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> samples)
            {
                Samples = samples;
            }
        }

        private Expr MakeSplitEntry(Function preFunc, Var[] outerInput, ImmutableArray<SingleSegment> infos)
        {
            var (newInputs, fixedShapeList) = Preprocess(preFunc, outerInput, infos);
            var innerFunc = InnerFnImpl(preFunc, fixedShapeList, infos);
            var then = new Call(innerFunc, newInputs);
            UpdateCaliData(preFunc, infos, innerFunc);
            return PostProcess(then, outerInput);
        }

        public class VarEqualityComparer : IEqualityComparer<Var>
        {
            public bool Equals(Var x, Var y)
            {
                return x.GlobalVarIndex == y.GlobalVarIndex;
            }

            public int GetHashCode(Var obj)
            {
                return obj.GetHashCode();
            }
        }

        private void UpdateCaliData(Function preFunc, ImmutableArray<SingleSegment> infos, Function innerFunc)
        {
            var vars = innerFunc.Parameters.ToArray();
            if (_options.QuantizeOptions.ModelQuantMode != ModelQuantMode.UsePTQ)
            {
                return;
            }
            var oldSamples = _options.QuantizeOptions.CalibrationDataset!.Samples;
            // var newSamples = oldSamples.ToEnumerable().Select(
            //     sample =>
            //     {
            //         var values = sample.Values;
            //         var (newCalibData, _) =
            //             Preprocess(preFunc, values.Select(v => (Expr)v.AsTensor()).ToArray(), infos);
            //         var newInputData = newCalibData.Select(expr => expr.Evaluate().AsTensor()).Select(t => (IValue)Value.FromTensor(t));
            //         var dict = vars.Zip(newInputData)
            //             .ToDictionary(pair => pair.Item1, pair => pair.Item2).Concat(sample)
            //             .ToDictionary(pair => pair.Key, pair => pair.Value);
            //         return dict;
            //     }
            // ).ToAsyncEnumerable();

            // _options.QuantizeOptions.CalibrationDataset = new DatasetProvider(newSamples);
        }

        public Expr PostProcess(Expr output, Var[] outerInput)
        {
            var actualLen = ShapeOf(outerInput[0])[1];
            var len = Cast(actualLen, DataTypes.Int32);
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
            dims[dimIndex] = seg;
            return dims.Select(x => x.FixedValue).ToArray();
        }

        private static Expr FitFixedShape(Expr targetInput, int[] fixedShape, int inputIndex)
        {
            targetInput.InferenceType();
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
                        preFunc.Name + $"_seg_{segStr}_inner_var_{index++}",
                        new TensorType(oldVar.CheckedDataType, fixedShape)));
                })
                .ToArray();

            var otherVar = preFunc.Parameters.ToArray().Where((_, i) => !inputIndex.Contains(i)).ToArray();
            var newOtherVar = otherVar.Select(v =>
                (v, new Var(preFunc.Name + $"_seg_{segStr}_inner_var_{index++}", v.TypeAnnotation)));
            var newBody = ReplaceVarInFunction(preFunc, innerFixedShapeVarList.Concat(newOtherVar).ToArray(), out var newParams);

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
