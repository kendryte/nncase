// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using System.Xml;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.Passes.Rules.ShapeBucket.ShapeBucketHelper;
using static Nncase.Passes.Rules.ShapeBucket.ShapeBucketRegister;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.ShapeBucket;

public sealed class SplitLLMStage : ModulePass
{
    public SplitLLMStage(CompileOptions compileOptions)
    {
        CompileOptions = compileOptions;
    }

    public CompileOptions CompileOptions { get; }

    protected override async Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        if (IsLLMMode(CompileOptions.ShapeBucketOptions) && input.Functions.Count == 1)
        {
            var entry = (Function)input.Entry!;
            Function prefill = await PermformAsync(entry, true, CompileSession, CompileOptions);
            Function decode = await PermformAsync(entry, false, CompileSession, CompileOptions);
            input.Add(prefill);
            input.Add(decode);
            Expr newBody;
            {
                var kvShape = IR.F.Tensors.ShapeOf(entry.Parameters[3]); // %past_key_values: f32[24,2,1,?,2,64]
                var kvLen = IR.F.Tensors.GetItem(kvShape, 3);
                var cond = IR.F.Math.Equal(kvLen, 0L);
                newBody = new IR.If(cond, new Call(prefill, entry.Parameters.ToArray()), new Call(decode, entry.Parameters.ToArray()));
            }

            input.Replace(0, entry.With(body: newBody));
        }

        return input;
    }

    private static async Task<Function> PermformAsync(Function func, bool prefill, CompileSession parentSession, CompileOptions compileOptions)
    {
        var newBucketOption = ShapeBucketOptions.CloneFrom(compileOptions.ShapeBucketOptions);
        if (prefill)
        {
            newBucketOption.RangeInfo.Remove("history_len");
            newBucketOption.FixVarMap.Add("history_len", 0);
        }
        else
        {
            newBucketOption.RangeInfo.Remove("seq_len");
            newBucketOption.FixVarMap.Add("seq_len", 1);
        }

        var vmap = func.Parameters.AsValueEnumerable().ToDictionary(v => v, v =>
        {
            // var newTypeHint = v.
            IRType typeAnnotation = v.TypeAnnotation;
            if (typeAnnotation is TensorType tensorType)
            {
                var fixedShape = tensorType.Shape.Select((s, i) =>
                    {
                        return s switch
                        {
                            { IsUnknown: true } when func.VarMap![v][i] is Var dimv && newBucketOption.FixVarMap.TryGetValue(dimv.Name, out var value) => value,
                            _ => s,
                        };
                    }).ToArray();
                typeAnnotation = tensorType with { Shape = fixedShape };
            }

            return v.With(v.Name + "_n", typeAnnotation);
        });
        var nParam = func.Parameters.AsValueEnumerable().Select(v => vmap[v]).ToArray();
        var nVarMap = func.Parameters.AsValueEnumerable().ToDictionary(v => vmap[v], v => func.VarMap![v]);
        newBucketOption.VarMap.Clear();
        foreach (var (k, shapeExprs) in nVarMap)
        {
            var nShapeExprs = new Expr[shapeExprs.Length];
            for (int i = 0; i < shapeExprs.Length; i++)
            {
                if (shapeExprs[i] is IR.Var dimv && newBucketOption.FixVarMap.TryGetValue(dimv.Name, out var value))
                {
                    nShapeExprs[i] = (long)value;
                }
                else
                {
                    nShapeExprs[i] = shapeExprs[i];
                }
            }

            newBucketOption.VarMap.Add(k, nShapeExprs);
        }

        var newCompileOptions = compileOptions with { ShapeBucketOptions = newBucketOption };
        var newSession = CompileSession.Create(parentSession.Target, newCompileOptions);
        using var sessionScope = new CompileSessionScope(newSession);
        var pmgr = newSession.CreatePassManager("pmgr_" + (prefill ? "prefill" : "decode"));
        RegisterBucketPass(pmgr, true);

        var cloner = new LLMCloner(vmap);
        var nBody = cloner.Clone(func.Body, default);
        var pre = new Function(prefill ? "prefill" : "decode", nBody, nParam, nVarMap);
        var subModule = new IR.IRModule(pre);
        var bucketed = await pmgr.RunAsync(subModule);
        return (Function)bucketed.Entry!;
    }

    private static void RegisterBucketPass(IPassManager p, bool singleVar)
    {
        ToFusion(p);
        MergeOp(p, true);
        // LostToFusion(p, singleVar);
        // MergeOp(p, true);
        // ClearMarker(p);
        // MergeFusion(p, singleVar, true);
        // Rebuild(p, singleVar);
        Bucket(p);
        Simplify(p);
    }
}

internal sealed class LLMCloner : ExprCloner<Unit>
{
    public LLMCloner(Dictionary<Var, Var> vmap)
    {
        Vmap = vmap;
    }

    public Dictionary<Var, Var> Vmap { get; }

    protected override Expr VisitLeafConst(Const expr, Unit context) => expr;

    protected override Expr VisitLeafVar(Var expr, Unit context)
    {
        if (Vmap.TryGetValue(expr, out var newV))
        {
            return newV;
        }

        return base.VisitLeafVar(expr, context);
    }
}
