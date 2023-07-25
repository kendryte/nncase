// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes.Analysis;
using Nncase.PatternMatch;
using Nncase.TIR;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class PrimFuncMergeRule : RewriteRule<PatternMatch.Pattern>
{
    private readonly HashSet<IR.BaseFunction> _mergedFuncs;

    public PrimFuncMergeRule(HashSet<IR.BaseFunction> mergedFuncs)
    {
        _mergedFuncs = mergedFuncs;
    }

    public override Pattern Pattern { get; } = IsCall("caller", IsWildcard("callerWrapper", e => e is PrimFunctionWrapper), IsVArgsRepeat("callerParams", (ReadOnlySpan<Expr> exprs) =>
    {
        var patterns = new List<Pattern>();
        for (int i = 0; i < exprs.Length; i++)
        {
            patterns.Add(IsAlt($"callerArgs{i}", IsCallWildcard($"callee_{i}", IsWildcard($"calleeWrapper_{i}", e => e is PrimFunctionWrapper)), IsWildcard()));
        }

        return patterns;
    }));

    private Expr? GetReplace(Call caller, PrimFunctionWrapper callerWrapper, IReadOnlyList<Expr> callerParams, IMatchResult result, RunPassContext context)
    {
        var userAnalysis = context.GetAnalysis<IExprUserAnalysisResult>();

        // foreach the callerParams try to merge.
        for (int i = 0; i < callerParams.Count; i++)
        {
            if (result[$"callerArgs{i}"] is not Call { Target: PrimFunctionWrapper })
            {
                continue;
            }

            var callee = (Call)result[$"callee_{i}"];
            var calleeWrapper = (PrimFunctionWrapper)result[$"calleeWrapper_{i}"];
            var calleeParams = (IReadOnlyList<Expr>)result[$"callee_{i}Params"];

            // when callee used by other caller, give up merge.
            if (userAnalysis[callee].Except(new[] { caller }).Any())
            {
                continue;
            }

            var mergeCall = Process(caller, callerWrapper, callerParams, callee, calleeWrapper, calleeParams);

            if (mergeCall is null)
            {
                continue;
            }

            return mergeCall;
        }

        return null;
    }

    private Expr? Process(Call caller, PrimFunctionWrapper callerWrapper, IReadOnlyList<Expr> callerParams, Call callee, PrimFunctionWrapper calleeWrapper, IReadOnlyList<Expr> calleeParams)
    {
        var calleeFunc = calleeWrapper.Target;
        var callerFunc = callerWrapper.Target;

        if (calleeFunc.ModuleKind != callerFunc.ModuleKind)
        {
            return null;
        }

        // 1. find the callee buffer index.
        List<int> calleeBufferIndexs = new();
        for (int i = 0; i < callerParams.Count; i++)
        {
            if (ReferenceEquals(callerParams[i], callee))
            {
                calleeBufferIndexs.Add(i);
            }
        }

        // 2. chack and create the data buffer
        if (calleeFunc.Parameters.ToArray().Count(b => b.MemLocation == MemoryLocation.Output) != 1)
        {
            // the direct call mean the callee function only have one output.
            return null;
        }

        var calleeRetBuffer = calleeFunc.Parameters[^1];
        var callerInBuffer = callerFunc.Parameters[calleeBufferIndexs[0]];
        if (!BufferCanMerge(calleeRetBuffer, callerInBuffer, out var dataBuffer))
        {
            return null;
        }

        // 3. merge the prim func body. skip the callee func end.
        var newBody = Sequential.Flatten(new object[] { calleeFunc.Body, callerFunc.Body });
        newBody = new PrimFuncCloner(new Dictionary<PhysicalBuffer, PhysicalBuffer>(ReferenceEqualityComparer.Instance) {
          { calleeRetBuffer, dataBuffer },
          { callerInBuffer, dataBuffer },
         }).Clone<Sequential>(newBody, default);

        // 4. build the new prim func.
        var newFuncParams = new List<TIR.PhysicalBuffer>();
        newFuncParams.AddRange(callerFunc.Parameters[..calleeBufferIndexs[0]].ToArray());
        newFuncParams.AddRange(calleeFunc.Parameters[..^1].ToArray());
        newFuncParams.AddRange(Enumerable.Range(0, callerFunc.Parameters.Length).Where(i => i > calleeBufferIndexs[0] && !calleeBufferIndexs.Contains(i)).Select(i => callerFunc.Parameters[i]).ToArray());
        var nameFunc = callerFunc.Name; // + '_' + calleeFunc.Name;
        var newFunc = new PrimFunction(nameFunc, calleeFunc.ModuleKind, newBody, newFuncParams.ToArray());

        // 5. build the new call.
        var nameWrapper = callerWrapper.Name; // + '_' + calleeWrapper.Name;
        var newWrapper = new PrimFunctionWrapper(nameWrapper, newFunc, newFuncParams.Count(b => b.MemLocation == MemoryLocation.Input));

        var newCallParams = new List<Expr>();
        newCallParams.AddRange(callerParams.Take(calleeBufferIndexs[0]));
        newCallParams.AddRange(calleeParams);
        newCallParams.AddRange(Enumerable.Range(0, callerParams.Count).Where(i => i > calleeBufferIndexs[0] && !calleeBufferIndexs.Contains(i)).Select(i => callerParams[i]).ToArray());

        // 6. add the merged in cache.
        _mergedFuncs.Add(calleeFunc);
        _mergedFuncs.Add(calleeWrapper);
        _mergedFuncs.Add(callerFunc);
        _mergedFuncs.Add(callerWrapper);

        return new Call(newWrapper, newCallParams.ToArray());
    }

    private bool BufferCanMerge(TIR.PhysicalBuffer retBuffer, TIR.PhysicalBuffer inBuffer, [MaybeNullWhen(false)] out TIR.PhysicalBuffer dataBuffer)
    {
        dataBuffer = null!;
        if (retBuffer.FixedDimensions.SequenceEqual(inBuffer.FixedDimensions) &&
            retBuffer.FixedStrides.SequenceEqual(inBuffer.FixedStrides) &&
            retBuffer.ElemType == inBuffer.ElemType &&
            retBuffer.Size == inBuffer.Size &&
            retBuffer.MemLocation == MemoryLocation.Output &&
            inBuffer.MemLocation == MemoryLocation.Input)
        {
            dataBuffer = new TIR.PhysicalBuffer(inBuffer.Name, inBuffer.ElemType, MemoryLocation.Data, inBuffer.FixedDimensions, inBuffer.FixedStrides, inBuffer.Start, inBuffer.Size);
            return true;
        }

        return false;
    }

    internal sealed class PrimFuncCloner : ExprCloner<Unit>
    {
        private readonly Dictionary<TIR.PhysicalBuffer, TIR.PhysicalBuffer> _dict;

        public PrimFuncCloner(Dictionary<TIR.PhysicalBuffer, TIR.PhysicalBuffer> dict)
        {
            _dict = dict;
        }

        protected override Expr VisitLeafLogicalBuffer(LogicalBuffer buffer, Unit context)
        {
            return buffer;
        }

        protected override Expr VisitLeafPhysicalBuffer(PhysicalBuffer buffer, Unit context)
        {
            if (_dict.TryGetValue(buffer, out var newBuffer))
            {
                return newBuffer;
            }

            return buffer;
        }

        protected override Expr VisitVar(Var var, Unit context)
        {
            return var;
        }
    }
}
