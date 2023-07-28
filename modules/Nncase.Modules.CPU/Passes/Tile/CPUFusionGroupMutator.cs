// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Runtime.CompilerServices;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Tile;

internal sealed class CPUSameInputFusionMergeRule : Mutators.SameInputFusionMergeRule
{
    public override string ModuleKind => CPUTarget.Kind;

    // todo enable multi input fusion merge pattern.
    public override Pattern CreatePattern(string target_module_kind)
    {
        var inputPat = IsWildcard("input");

        var callerPattern = IsCall(
            "caller",
            IsFusion(
                "caller_fusion",
                target_module_kind,
                IsWildcard(),
                IsVArgs(IsWildcard())),
            IsVArgs("caller_inputs", new[] {
              IsCall(
                  $"callee_{0}",
                  IsFusion($"callee_fusion_{0}", target_module_kind, IsWildcard(), IsVArgs(IsWildcard())),
                  inputPat),
            }));
        return callerPattern;
    }
}

internal sealed class CPUFusionGroupMutator<T> : Mutators.FusionGroupMutator
  where T : IFusionChecker
{
    private readonly TileOptions _tileOptions;

    public CPUFusionGroupMutator(
      Dictionary<Fusion, IFusionChecker> fusioncheckerCache,
      TileOptions tileOptions,
      Mutators.IMergeRewriteRule rule,
      RunPassContext passOptions)
        : base(rule, passOptions)
    {
        _tileOptions = tileOptions;
        FusioncheckerCache = fusioncheckerCache;
    }

    public Dictionary<Fusion, IFusionChecker> FusioncheckerCache { get; }

    /// <inheritdoc/>
    public override bool MergedFusionCheckCallBack(Fusion mergedFusion, HashSet<Fusion> candidateFusions)
    {
        // var checker = (IFusionChecker)Activator.CreateInstance(typeof(T), new object[] { _tileOptions })!;
        // var ret = checker.Check(mergedFusion, PassOptions);
        // if (ret)
        // {
        //     FusioncheckerCache.Add(mergedFusion, checker);
        //     foreach (var cand in candidateFusions)
        //     { // release the merged fusion.
        //         FusioncheckerCache.Remove(cand);
        //     }
        // }

        // return ret;
        return false;
    }

    public override Expr MergedFusionRewriteCallBack(Expr mergedFusionBody)
    {
        return mergedFusionBody;
    }
}

internal sealed class CheckedConvertMutator : ExprRewriter
{
    private readonly Dictionary<Fusion, BaseFunction> _fusionConertedCache;
    private readonly IReadOnlyDictionary<Fusion, IFusionChecker> _fusionCheckerCache;
    private readonly TileOptions _tileOptions;
    private readonly RunPassContext _passOptions;

    public CheckedConvertMutator(Dictionary<Fusion, BaseFunction> fusion_converted_cache, IReadOnlyDictionary<Fusion, IFusionChecker> fusionchecker_cache, TileOptions tileOptions, RunPassContext passOptions)
    {
        _fusionConertedCache = fusion_converted_cache;
        _fusionCheckerCache = fusionchecker_cache;
        _tileOptions = tileOptions;
        _passOptions = passOptions;
    }

    /// <inheritdoc/>
    protected override Expr RewriteLeafFusion(Fusion expr)
    {
        if (expr is Fusion { ModuleKind: CPUTarget.Kind } fusion)
        {
            if (!_fusionConertedCache.TryGetValue(fusion, out _))
            {
                TIR.PrimFunction prim_func;
                if (_fusionCheckerCache.TryGetValue(fusion, out var checker))
                {
                    prim_func = checker.Convert(_passOptions);
                }
                else
                {
                    var converter = new SingleCPUFusionConverter();
                    prim_func = converter.Visit(fusion);
                }

                BaseFunction? convert_func = prim_func;
                _fusionConertedCache.Add(fusion, convert_func);
            }
        }

        return expr;
    }

    protected override Expr RewriteLeafCall(Call expr)
    {
        if (expr.Target is Fusion { ModuleKind: CPUTarget.Kind } fusion)
        {
            var convert_func = _fusionConertedCache[fusion];
            PrimFunctionWrapper wrapper;
            if (convert_func is TIR.PrimFunction prim_func)
            {
                bool is_input = true;
                int param_count = 0;
                foreach (var b in prim_func.Parameters)
                {
                    if (b.MemSpan.Location == TIR.MemoryLocation.Input)
                    {
                        if (is_input)
                        {
                            param_count += 1;
                        }
                        else
                        {
                            throw new InvalidOperationException("The output buffer must behind the input buffer");
                        }
                    }
                    else
                    {
                        is_input = false;
                    }
                }

                wrapper = new PrimFunctionWrapper(prim_func, param_count);
                _fusionConertedCache[fusion] = wrapper;
            }
            else
            {
                wrapper = (PrimFunctionWrapper)convert_func;
            }

            return expr.With(target: wrapper);
        }

        return expr;
    }
}
