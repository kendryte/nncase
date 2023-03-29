// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.Passes.Analysis;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Mutators;

/// <summary>
/// the fusion group mutator.
/// </summary>
public class FusionGroupMutator : ExprRewriter
{
    private readonly IExprUserAnalysisResult _userAnalysis;

    /// <summary>
    /// cache the check result.
    /// </summary>
    private readonly Dictionary<HashSet<Fusion>, bool> _candidateFusionCache;

    /// <summary>
    /// Initializes a new instance of the <see cref="FusionGroupMutator"/> class.
    /// ctor.
    /// </summary>
    /// <param name="fusionRule">pre order rule. </param>
    /// <param name="passOptions">pass options. </param>
    public FusionGroupMutator(IMergeRewriteRule fusionRule, RunPassContext passOptions)
    {
        passOptions.GetAnalysis(out _userAnalysis);
        Rule = fusionRule;
        PassOptions = passOptions;
        _candidateFusionCache = new(new FusionMergeCandidateComparer());
    }

    /// <summary>
    /// Gets the Pre Order Rules.
    /// </summary>
    public IMergeRewriteRule Rule { get; }

    /// <summary>
    /// Gets get the merge check cache result.
    /// </summary>
    public IReadOnlyDictionary<HashSet<Fusion>, bool> FusionMergeCandidateCache => _candidateFusionCache;

    protected RunPassContext PassOptions { get; }

    /// <summary>
    /// check the merged fusion is valid.
    /// </summary>
    /// <param name="merged_fusion">merged fusion.</param>
    /// <param name="candidate_fusions">candidate fusions.</param>
    /// <returns>bool.</returns>
    public virtual bool MergedFusionCheckCallBack(Fusion merged_fusion, HashSet<Fusion> candidate_fusions)
    {
        return true;
    }

    /// <summary>
    /// when fusion merged, maybe need rewrite somethings.
    /// </summary>
    /// <param name="merged_fusion_body">merged fusion body.</param>
    /// <returns>rewrited body.</returns>
    public virtual Expr MergedFusionRewriteCallBack(Expr merged_fusion_body)
    {
        return merged_fusion_body;
    }

    /// <summary>
    /// try merge fusion from the old call.
    /// </summary>
    /// <param name="rule">rule.</param>
    /// <param name="old_call">current call.</param>
    /// <param name="new_call">returned new call.</param>
    /// <returns>merged status. </returns>
    public bool TryMergeFusion(IMergeRewriteRule rule, Call old_call, out Call new_call)
    {
        new_call = null!;

        if (!CompilerServices.TryMatchRoot(old_call, rule.Pattern, new(), out var result))
        {
            return false;
        }

        if (rule.GetReplace(
            MergedFusionRewriteCallBack,
            MergedFusionCheckCallBack,
            CandidateFusionCheckCallBack,
            CandidateFusionRecordCallBack,
            _userAnalysis,
            result,
            PassOptions) is Call replaced_call)
        {
            new_call = replaced_call;
            return true;
        }

        return false;
    }

    protected override Expr VisitFusion(Fusion expr, Unit context) => base.VisitFusion(expr, context);

    /// <inheritdoc/>
    protected override Expr RewriteLeafCall(Call expr)
    {
        // note only rewrite once. avoid RAUW problem.
        if (!IsMutated && TryMergeFusion(Rule, expr, out var merged_call))
        {
            return merged_call;
        }

        return expr;
    }

    private bool CandidateFusionCheckCallBack(HashSet<Fusion> candidateFusions)
    {
        if (candidateFusions.Count <= 1)
        {
            throw new InvalidDataException("The candidates less than 2!");
        }

        if (!_candidateFusionCache.TryGetValue(candidateFusions, out var ret))
        {
            return true;
        }

        if (ret != false)
        {
            throw new InvalidDataException("Only cache failed candidates!");
        }

        return false;
    }

    private void CandidateFusionRecordCallBack(HashSet<Fusion> candidateFusions)
    {
        if (candidateFusions.Count <= 1)
        {
            throw new InvalidDataException("The candidates less than 2!");
        }

        _candidateFusionCache.Add(candidateFusions, false);
    }

    private sealed class FusionMergeCandidateComparer : IEqualityComparer<HashSet<Fusion>>
    {
        public bool Equals(HashSet<Fusion>? x, HashSet<Fusion>? y) => (x, y) switch
        {
            (null, null) => true,
            (null, _) => false,
            (_, null) => false,
            (var lhs, var rhs) => GetHashCode(lhs) == GetHashCode(rhs),
        };

        public int GetHashCode([DisallowNull] HashSet<Fusion> obj)
        {
            var hash = default(HashCode);
            foreach (var o in obj)
            {
                hash.Add(ReferenceEqualityComparer.Instance.GetHashCode(obj));
            }

            return hash.ToHashCode();
        }
    }
}
