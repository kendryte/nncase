// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Distributed;

public sealed partial class AutoDistributedWithShapeBucketPass : FunctionPass
{
    private const int LargeTensorSizeThreshold = 1000; // Threshold for large tensors in bytes

    private readonly CompileOptions _compileOptions;

    private readonly bool _bidirectional;

    private readonly string _moduleKind;

    private int _bufferIndex;

    public AutoDistributedWithShapeBucketPass(bool bidirectional, string moduleKind, CompileOptions compileOptions)
    {
        _compileOptions = compileOptions;
        _bidirectional = bidirectional;
        _moduleKind = moduleKind;
    }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function function || input.Metadata is AutoDistributedMetaData { Skip: true })
        {
            return Task.FromResult(input);
        }

        if (_compileOptions.TargetOptions is INTTTargetOptions targetOptions)
        {
            var newFunction = Distribute(function, targetOptions);
            return Task.FromResult((BaseFunction)newFunction);
        }

        return Task.FromResult(input);
    }

    private PrimFunction Distribute(Function function, INTTTargetOptions targetOptions)
    {
        var rewriter = new AutoDistributedRewriter(_compileOptions, targetOptions, AutoDistributedPhase.SearchConstant, _moduleKind, _bidirectional);
        rewriter.Rewrite(function);

        var distributedConsts = rewriter.DistributedConsts;
        var functionWithDistributedConsts = new DistributeConstCloner(distributedConsts).Clone(function, Unit.Default);

        var dumpper = DumpScope.Current;
        if (dumpper.IsEnabled(DumpFlags.PassIR))
        {
            dumpper.DumpIR(functionWithDistributedConsts, "FunctionWithDistributedConsts");
        }

        var shapeBucketOptions = _compileOptions.ShapeBucketOptions;
        var segmentFunctions = new List<(Function SegmentFunction, Dictionary<DimVar, DimVar> DimVars)>();
        for (int segmentIndex = 0; segmentIndex < shapeBucketOptions.SegmentsCount; segmentIndex++)
        {
            var newDimVars = (from dimVar in shapeBucketOptions.VarMap.Keys.OfType<DimVar>()
                              let newRange = ShapeUtility.GetDimSegmentRange(dimVar.Metadata.Range!.Value, segmentIndex, shapeBucketOptions.SegmentsCount)
                              select new KeyValuePair<DimVar, DimVar>(
                                  dimVar,
                                  dimVar.With(range: newRange))).ToDictionary(kvp => kvp.Key, kvp => kvp.Value, (IEqualityComparer<DimVar>)ReferenceEqualityComparer.Instance);
            var segmentFunction = new SegmentFunctionCloner(newDimVars).Clone(functionWithDistributedConsts, Unit.Default)
                .With(name: $"{function.Name}_segment_{segmentIndex}");
            rewriter = new AutoDistributedRewriter(_compileOptions, targetOptions, AutoDistributedPhase.Final, _moduleKind, _bidirectional);
            segmentFunctions.Add((rewriter.Rewrite(segmentFunction), newDimVars));
        }

        return BuildMainFunction(function, segmentFunctions);
    }

    private PrimFunction BuildMainFunction(Function inputFunction, List<(Function SegmentFunction, Dictionary<DimVar, DimVar> DimVars)> segmentFunctions)
    {
        var outputBuffers = CreateOutputBuffers(inputFunction.Body);
        var inputParams = inputFunction.Parameters.AsValueEnumerable().Select(x => (Expr)x).ToArray().Concat(outputBuffers).ToArray();
        TIR.Sequential MakeSegementCall(Function segmentFunction)
        {
            return T.Sequential()
                .Body(new Call(new FunctionWrapper(segmentFunction.ModuleKind, segmentFunction), inputParams))
                .Build();
        }

        Expr lastSegmentCall = MakeSegementCall(segmentFunctions[^1].SegmentFunction);
        for (int i = segmentFunctions.Count - 2; i >= 0; i--)
        {
            var dimVars = segmentFunctions[i].DimVars;
            var segmentCall = MakeSegementCall(segmentFunctions[i].SegmentFunction);
            var condition = dimVars
                .Select(dimVarPair => dimVarPair.Key <= (long)dimVarPair.Value.Metadata.Range!.Value.Max)
                .Aggregate(IR.F.Math.LogicalAnd);
            lastSegmentCall = T.If(condition)
                .Then(segmentCall)
                .Else(lastSegmentCall)
                .Build();
        }

        var mainBody = T.Sequential()
            .Body(outputBuffers)
            .Body(
                lastSegmentCall,
                T.Return(outputBuffers))
            .Build();
        return new PrimFunction($"{inputFunction.Name}_prim", inputFunction.ModuleKind, mainBody, inputFunction.Parameters) { Metadata = inputFunction.Metadata };
    }

    private TIR.Buffer[] CreateOutputBuffers(BaseExpr expr)
    {
        var memoryLocation = MemoryLocation.Output;
        if (expr.CheckedType is TupleType tt)
        {
            var fields = tt.Fields.AsValueEnumerable().Select(x => CreateBuffer(x, memoryLocation)).ToArray();
            return fields;
        }
        else
        {
            return [CreateBuffer(expr.CheckedType, memoryLocation)];
        }
    }

    private TIR.Buffer CreateBuffer(IRType type, MemoryLocation memoryLocation)
    {
        var tensorType = type switch
        {
            DistributedType dt => dt.TensorType,
            TensorType tt => tt,
            _ => throw new ArgumentException($"Unsupported type: {type}"),
        };
        return T.CreateBuffer(tensorType, memoryLocation, out _, $"buffer_{_bufferIndex++}", type as DistributedType);
    }

    private sealed class DistributeConstCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<TensorConst, TensorConst> _distributedConsts;

        public DistributeConstCloner(IReadOnlyDictionary<TensorConst, TensorConst> distributedConsts)
        {
            _distributedConsts = distributedConsts;
            CloneUnmutated = false;
        }

        protected override BaseExpr VisitLeafTensorConst(TensorConst expr, Unit context)
        {
            if (_distributedConsts.TryGetValue(expr, out var distConst) &&
                expr.Value.Length * expr.Value.ElementType.SizeInBytes > LargeTensorSizeThreshold)
            {
                // If the tensor is large, we do not remove boxing
                return IR.F.Distributed.Boxing(distConst, expr.CheckedTensorType);
            }

            return expr;
        }
    }

    private sealed class SegmentFunctionCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<DimVar, DimVar> _newDimVars;

        public SegmentFunctionCloner(IReadOnlyDictionary<DimVar, DimVar> newDimVars)
        {
            _newDimVars = newDimVars;
            CloneUnmutated = false;
        }

        protected override BaseExpr VisitLeafVar(Var expr, Unit context)
        {
            bool IsOperandsMutated()
            {
                return IsMutatedType(expr.TypeAnnotation, context);
            }

            if (CloneUnmutated || IsOperandsMutated())
            {
                return expr.With(
                    typeAnnotation: CloneType(expr.TypeAnnotation, context));
            }

            return expr;
        }

        protected override BaseExpr VisitLeafDimVar(DimVar expr, Unit context)
        {
            return _newDimVars.GetValueOrDefault(expr, expr);
        }
    }
}
