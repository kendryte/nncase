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
        private Expr Default => (Expr)IR.F.Math.Require(false, 0, "input dim large than limit");

        public IRModule Run(Function preFunc, SegmentInfo info)
        {
            return null;
        }

        private record SingleSegment(int InputIndex, int DimIndex, int SegmentLimit)
        {
        }

        public IRModule SplitInfos(Function preFn, SegmentInfo[] infos)
        {
            var body = SplitImpl(preFn, infos, MakeSplitEntry(preFn));
            var newFn = new Function("main", body, preFn.Parameters);

            return CollectFunctionToNewModule(newFn);
        }

        private Expr SplitImpl(Function preFunc, SegmentInfo[] infos,
            Func<ImmutableArray<SingleSegment>, Expr> thenCtor)
        {
            var segments = Array.Empty<SingleSegment>().ToImmutableArray();
            return SplitImpl(preFunc, infos, 0, infos.Length, segments, thenCtor);
        }

        private Expr SplitImpl(Function preFunc, SegmentInfo[] infos, int current, int limit,
            ImmutableArray<SingleSegment> segments, Func<ImmutableArray<SingleSegment>, Expr> thenCtor)
        {
            var info = infos[current];
            Check(preFunc, info);

            // todo: segments sort
            var body = info.Segments.OrderBy(x => x).Aggregate(
                Default,
                (sum, seg) =>
                {
                    var inShape = ShapeOf(preFunc.Parameters[info.InputIndex]);
                    var dim = Cast(inShape, DataTypes.Int32)[info.DimIndex];
                    var single = new SingleSegment(info.InputIndex, info.DimIndex, seg);
                    var newSegments = segments.Append(single).ToImmutableArray();
                    var then = current + 1 < limit
                        ? SplitImpl(preFunc, infos, current + 1, limit, newSegments, thenCtor)
                        : thenCtor(newSegments);
                    return new If(
                        dim <= seg,
                        then,
                        sum);
                });
            return body;
        }

        private static void Check(Function f, SegmentInfo info)
        {
            Debug.Assert(info.Segments.Length >= 2, "Segments.Length >= 2");
            Debug.Assert(f.Parameters.Count >= info.InputIndex, "f.Parameters.Count <= info.InputIndex");
        }

        private Func<ImmutableArray<SingleSegment>, Expr> MakeSplitEntry(Function preFunc) => infos =>
        {
            var fixedShapeList = infos.Select(info =>
                ComputeFixedShape(preFunc, info.InputIndex, info.DimIndex, info.SegmentLimit)).ToArray();

            var fitFixedShape = infos.Zip(fixedShapeList)
                .Select(info => FitFixedShape(preFunc, info.First.InputIndex, info.Second))
                .ToArray();

            var wrapperFn = new Function(preFunc.Name + $"_{MakeSegmentsStr(infos)}_wrapper", new IR.Tuple(fitFixedShape), preFunc.Parameters);
            var fitFixedShapeInputs = new Call(wrapperFn, preFunc.Parameters.ToArray());

            var newInputs = preFunc.Parameters.Select(v => (Expr)v).ToArray();
            for (int i = 0; i < infos.Length; i++)
            {
                newInputs[infos[i].InputIndex] = fitFixedShapeInputs[i];
            }

            var innerFunc = SplitFuncImpl(preFunc, fixedShapeList, infos);
            var then = new Call(innerFunc, newInputs);
            return then;
        };

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

        private static int[] ComputeFixedShape(Function preFunc, int inputIndex, int dimIndex, int seg)
        {
            var originShape = preFunc.Parameters[inputIndex].CheckedShape;
            var dims = originShape.Select(x => x).ToArray();

            // only dims[DimIndex] is unknown
            dims[dimIndex] = seg;
            return dims.Select(x => x.FixedValue).ToArray();
        }

        private static Expr FitFixedShape(Function preFunc, int inputIndex, int[] fixedShape)
        {
            // rename
            var wrapParams = preFunc.Parameters.ToArray();
            wrapParams[inputIndex] = wrapParams[inputIndex] with { Name = "split_dynamic_input" };
            var targetInput = wrapParams[inputIndex];

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

        private static string MakeSegmentsStr(ImmutableArray<SingleSegment> segments) => string.Join("_", segments.Select(x => x.SegmentLimit));

        private Function SplitFuncImpl(Function preFunc, int[][] fixedShapeList, ImmutableArray<SingleSegment> segments)
        {
            var segStr = MakeSegmentsStr(segments);
            var inputIndex = segments.Select(s => s.InputIndex);
            var innerFixedShapeVarList = fixedShapeList
                .Zip(inputIndex)
                .Select(indexAndShape =>
                {
                    var (fixedShape, inIndex) = indexAndShape;
                    var oldVar = preFunc.Parameters[inIndex];
                    return (oldVar, new Var(
                        preFunc.Name + $"_seg_{MakeSegmentsStr(segments)}_inner_var",
                        new TensorType(oldVar.CheckedDataType, fixedShape)));
                })
                .ToArray();

            var newBody = innerFixedShapeVarList.Aggregate(
                preFunc.Body,
                (tmpBody, vars) =>
                {
                    var (preVar, innerFixedShapeVar) = vars;
                    return ReplaceUtility.ReplaceExpr(tmpBody, preVar, innerFixedShapeVar);
                });

            // var newBody = ReplaceUtility.ReplaceExpr(preFunc.Body, preVar, innerFixedShapeVar);
            var newParams = ReplaceUtility.ReplaceItems(
                preFunc.Parameters,
                innerFixedShapeVarList
                    .Select(x => ((Expr)x.oldVar, (Expr)x.Item2))
                    .ToArray())
                .Select(x => (Var)x)
                .ToArray();

            // replace Fix var.
            var innerFunc = new Function(
                preFunc.Name + $"_seg_{segStr}_inner",
                newBody,
                newParams);
            return innerFunc;
        }
    }
}
