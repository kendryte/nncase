﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Targets;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.CPU;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.CPU;

[RuleGenerator]
internal sealed partial class CPUOutputBoxingFusion : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern { get; } = IsBoxing(
        target_name: "boxing",
        op => op.NewType is TensorType,
        IsCallWildcard("call", IsOp<Op>("op", _ => true)) with { TypePattern = IsType(t => t is DistributedType) });

    private Call? GetReplace(Call call, Op op, Boxing boxing, IReadOnlyList<Expr> callParams)
    {
        var newInputs = new List<Expr>();
        for (int i = 0; i < callParams.Count; i++)
        {
            if (callParams[i] is Call or Var)
            {
                newInputs.Add(new Var(callParams[i].CheckedType!));
            }
            else
            {
                if (callParams[i] is TensorConst { Value: Tensor { Shape.IsScalar: true } } tc)
                {
                    newInputs.Add(Const.FromTensor(Tensor.FromBytes(tc.CheckedDataType, tc.Value.BytesBuffer.ToArray(), new[] { 1 })));
                }
                else
                {
                    newInputs.Add(callParams[i]);
                }
            }
        }

        var newCall = new Call(op, newInputs.ToArray());
        var newBoxingCall = new Call(boxing, newCall);
        var callFusion = new Call(new Fusion($"{op.GetType().Name}_{Count++}_kernel", ModuleKind, newBoxingCall, newInputs.OfType<Var>().ToArray()), newInputs.Select((e, i) => (e, i)).Where(p => p.e is Var).Select(p => callParams[p.i]).ToArray());
        return callFusion;
    }
}

[RuleGenerator]
internal sealed partial class CPUSingleFusion : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern { get; } = IsCallWildcard(
        "call",
        IsOp<Op>("op", _ => true)) with { TypePattern = IsType(t => t is DistributedType) };

    private Call? GetReplace(Call call, Op op, IReadOnlyList<Expr> callParams)
    {
        var newInputs = new List<Expr>();
        for (int i = 0; i < callParams.Count; i++)
        {
            if (callParams[i] is Call or Var)
            {
                newInputs.Add(new Var(callParams[i].CheckedType!));
            }
            else
            {
                if (callParams[i] is TensorConst { Value: Tensor { Shape.IsScalar: true } } tc)
                {
                    newInputs.Add(Const.FromTensor(Tensor.FromBytes(tc.CheckedDataType, tc.Value.BytesBuffer.ToArray(), new[] { 1 })));
                }
                else
                {
                    newInputs.Add(callParams[i]);
                }
            }
        }

        var newCall = new Call(op, newInputs.ToArray());
        var callFusion = new Call(new Fusion($"{op.GetType().Name}_{Count++}_kernel", ModuleKind, newCall, newInputs.OfType<Var>().ToArray()), newInputs.Select((e, i) => (e, i)).Where(p => p.e is Var).Select(p => callParams[p.i]).ToArray());
        return callFusion;
    }
}

internal sealed class FusionCostEvaluator : Evaluator.IBaseFuncCostEvaluator
{
    private readonly CompileOptions _compileOptions;

    public FusionCostEvaluator(CompileOptions compileOptions)
    {
        _compileOptions = compileOptions;
    }

    public Cost VisitLeaf(IR.BaseFunction target)
    {
        if (target is Fusion fusion)
        {
            return new FusionGraphCostVisitor(_compileOptions).Visit(fusion);
        }
        else
        {
            throw new NotSupportedException();
        }
    }

    private sealed class GraphOpCostEvaluateContext : Evaluator.ICostEvaluateContext
    {
        private readonly IRType? _returnType;
        private readonly IRType?[] _argumentTypes;
        private readonly Expr[] _arguments;

        public GraphOpCostEvaluateContext(IRType? returnType, IRType?[] argumentTypes, ReadOnlySpan<Expr> arguments, CompileOptions compileOptions)
        {
            _returnType = returnType;
            _argumentTypes = argumentTypes;
            CompileOptions = compileOptions;
            _arguments = arguments.ToArray();
        }

        public CompileOptions CompileOptions { get; }

        public T GetArgument<T>(Op op, ParameterInfo parameter)
          where T : IR.BaseFunction
        {
            return (T)_arguments[parameter.Index];
        }

        public T GetArgumentType<T>(Op op, ParameterInfo parameter)
            where T : IRType
        {
            if (op.GetType() == parameter.OwnerType)
            {
                return (T?)_argumentTypes[parameter.Index] ?? throw new InvalidOperationException("Run type infer first.");
            }
            else
            {
                throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
            }
        }

        public T GetReturnType<T>()
            where T : IRType
        {
            return (T?)_returnType ?? throw new InvalidOperationException("Run type infer first.");
        }
    }

    private sealed class FusionGraphCostVisitor : ExprVisitor<Cost, IRType>
    {
        public FusionGraphCostVisitor(CompileOptions compileOptions)
        {
            CompileOptions = compileOptions;
        }

        public CompileOptions CompileOptions { get; }

        protected override Cost VisitLeafVar(Var var)
        {
            return new Cost()
            {
                [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(var.CheckedType!),
            };
        }

        protected override Cost DefaultVisitLeaf(Expr expr)
        {
            return Cost.Zero;
        }

        protected override Cost VisitLeafCall(Call call)
        {
            Cost cost;
            if (call.Target is Op op)
            {
                var context = new GraphOpCostEvaluateContext(call.CheckedType, call.Arguments.AsValueEnumerable().Select(p => p.CheckedType).ToArray(), call.Arguments, CompileOptions);
                cost = CompilerServices.EvaluateOpCost(op, context) ?? Cost.Zero;
            }
            else
            {
                throw new NotSupportedException();
            }

            return cost;
        }

        protected override Cost VisitLeafFusion(Fusion fusion)
        {
            var cost = new Cost()
            {
                [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(fusion.Body.CheckedType!),
            };
            cost += fusion.Parameters.AsValueEnumerable().Select(Visit).Sum() ?? Cost.Zero;
            return cost;
        }
    }
}

internal sealed class FusionMerger : ExprCloner<Unit>
{
    private readonly Dictionary<Var, Expr> _varMap;

    public FusionMerger(Dictionary<Var, Expr> varMap)
    {
        _varMap = varMap;
    }

    protected override Expr VisitLeafVar(Var v, Unit context)
    {
        if (_varMap.TryGetValue(v, out var new_expr))
        {
            return Visit(new_expr, context);
        }

        return v;
    }
}

internal sealed class GeneralFusionMergeRule : IRewriteRule
{
    private readonly Dictionary<int, Call> _mergedCache = new();

    public IPattern Pattern { get; } =
    IsCall(
        "caller",
        IsFusion("caller_fusion", _ => true, IsWildcard(), IsVArgsRepeat("inputs", exprs => {
            var patterns = new Pattern[exprs.Length];
            for (var i = 0; i < patterns.Length; i++)
            {
                patterns[i] = IsWildcard($"input_{i}");
            }

            return patterns;
        })),
        IsVArgsRepeat("callerInputs", exprs => {
            var patterns = new Pattern[exprs.Length];
            for (var i = 0; i < patterns.Length; i++)
            {
                patterns[i] = IsWildcard($"callee_{i}");
            }

            return patterns;
        }));

    public Expr? GetReplace(IMatchResult result, RunPassContext options)
    {
        var caller = (Call)result["caller"];
        var caller_fusion = (Fusion)result["caller_fusion"];
        var callerInputs = (IReadOnlyList<Expr>)result["callerInputs"];
        var callees = new List<Call>();
        var callee_fusions = new List<Fusion>();
        var fusion_index = new List<int>();
        for (var i = 0; i < callerInputs.Count; i++)
        {
            if (result[$"callee_{i}"] is Call { Target: Fusion } callee)
            {
                var callee_fusion = callee.Target as Fusion;
                if (callee_fusion!.ModuleKind == caller_fusion.ModuleKind)
                {
                    callees.Add(callee);
                    callee_fusions.Add(callee_fusion);
                    fusion_index.Add(i);
                }
            }
        }

        for (var i = callees.Count - 1; i >= 0; i--)
        {
            var callee = callees[i];
            var callee_fusion = callee_fusions[i];
            if (callees.Except(new[] { callee }).Any(c => c.Arguments.ToArray().Any(a => a == callee)))
            {
                callees.RemoveAt(i);
                callee_fusions.RemoveAt(i);
                fusion_index.RemoveAt(i);
            }
        }

        if (callees.Count == 0)
        {
            return null;
        }

        var hashCodes = new List<int>
        {
            ReferenceEqualityComparer.Instance.GetHashCode(caller_fusion),
        };
        foreach (var fusion in callee_fusions)
        {
            hashCodes.Add(ReferenceEqualityComparer.Instance.GetHashCode(fusion));
        }

        var hash = default(HashCode);
        foreach (var subHash in hashCodes)
        {
            hash.Add(subHash);
        }

        var hashcode = hash.ToHashCode();
        if (!_mergedCache.TryGetValue(hashcode, out var new_call))
        {
            var multiVarMap = new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance);
            for (var index = 0; index < fusion_index.Count; index++)
            {
                multiVarMap.Add(caller_fusion.Parameters[fusion_index[index]], callee_fusions[index].Body);
            }

            var new_fusion_body = new FusionMerger(multiVarMap).Clone(caller_fusion.Body, default);
            var name = $"fusion_{caller_fusion.Name}_" + string.Join("_", callee_fusions.Select(f => f.Name).ToArray());

            // remove duplicate callees
            var seen = new HashSet<Expr>();
            var remindIndex = Enumerable.Range(0, callerInputs.Count).ToList();
            for (var i = callees.Count - 1; i >= 0; i--)
            {
                if (!seen.Add(callees[i]))
                {
                    callees.RemoveAt(i);
                    callee_fusions.RemoveAt(i);
                    remindIndex.RemoveAt(fusion_index[i]);
                    fusion_index.RemoveAt(i);
                }
            }

            var parameters = remindIndex.Select(i => fusion_index.Contains(i) ? callee_fusions[fusion_index.IndexOf(i)].Parameters.ToArray() : new[] { caller_fusion.Parameters[i] }).SelectMany(e => e).ToArray();
            var merged_fusion = new Fusion(name, caller_fusion.ModuleKind, new_fusion_body, parameters);

            var calleeInputs = remindIndex.Select(i => fusion_index.Contains(i) ? callees[fusion_index.IndexOf(i)].Arguments.ToArray() : new[] { callerInputs[i] }).SelectMany(a => a).ToArray();
            new_call = new Call(merged_fusion, calleeInputs.ToArray());
            _mergedCache.Add(hashcode, new_call);
        }
        else
        {
            // System.Console.WriteLine("Re Add Merged Two Fusion Call");
        }

        return new_call;
    }
}