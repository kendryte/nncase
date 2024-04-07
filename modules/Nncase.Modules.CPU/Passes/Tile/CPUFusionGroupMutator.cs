// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Mutators;
using Nncase.Targets;

[assembly: InternalsVisibleTo("Nncase.Tests.CPU")]

namespace Nncase.Passes.Tile;

internal sealed class CPUSameInputFusionMergeRule : SameInputFusionMergeRule
{
    public override string ModuleKind => CPUTarget.Kind;
}

internal sealed class CPUMultiInputFusionMergeRule : MultiInputFusionMergeRule
{
    public override string ModuleKind => CPUTarget.Kind;
}

internal sealed class CPUShortCutFusionMergeRuleLeft : ShortCutFusionMergeRuleLeft
{
    public override string ModuleKind => CPUTarget.Kind;
}

internal sealed class CPUShortCutFusionMergeRuleRight : ShortCutFusionMergeRuleRight
{
    public override string ModuleKind => CPUTarget.Kind;
}

internal sealed class CPUFusionGroupMutator : FusionGroupMutator
{
    private readonly Dictionary<Fusion, FusionChecker> _fusioncheckerCache;
    private bool _checked;

    // private readonly TileOptions _tileOptions = null!;
    public CPUFusionGroupMutator(
        Dictionary<Fusion, FusionChecker> fusioncheckerCache,
        IMergeRewriteRule rule,
        RunPassContext passOptions)
        : base(rule, passOptions)
    {
        _fusioncheckerCache = fusioncheckerCache;
        _checked = false;
    }

    /// <inheritdoc/>
    public override bool MergedFusionCheckCallBack(Fusion mergedFusion, HashSet<Fusion> candidateFusions)
    {
        bool ok = false;
        if (!_checked)
        {
            PrimTileVisitor primTileVisitor = new();
            primTileVisitor.Visit(mergedFusion.Body);
            var checker = new FusionChecker(primTileVisitor.TileList);

            // CompilerServices.DumpDotIR(merged_fusion, "before_merge_check", PassOptions.DumpDir,true); // dump sub function.
            var ret = checker.Check(mergedFusion.Body);
            ok = ret.Count > 0;

            // CompilerServices.DumpDotIR(merged_fusion, "after_merge_check", PassOptions.DumpDir,true); // dump sub function.
            if (ok)
            {
                _checked = true;
                _fusioncheckerCache.Add(mergedFusion, checker);
                foreach (var cand in candidateFusions)
                {
                    // release the merged fusion.
                    _fusioncheckerCache.Remove(cand);
                }
            }
        }

        return ok;
    }

    public override Expr MergedFusionRewriteCallBack(Expr mergedFusionBody)
    {
        using var dumpScope = new DumpScope("MergedFusionClear");
        return CompilerServices.ERewrite(mergedFusionBody, new[] { new Passes.Rules.CPU.FoldStoreLoad() }, new());
    }

    protected override Expr RewriteLeafCall(Call expr)
    {
        return _checked ? expr : base.RewriteLeafCall(expr);
    }
}
