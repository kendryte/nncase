// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Mutators;

/// <summary>
/// the fusion group mutator.
/// </summary>
public class FusionGroupMutator : ExprMutator
{
    /// <summary>
    /// Get the run pass options.
    /// </summary>
    public readonly RunPassContext PassOptions;

    /// <summary>
    /// Get the Pre Order Rules.
    /// </summary>
    public readonly IMergeRewriteRule Rule;

    private readonly IUsedByResult _usedByReslut;

    /// <summary>
    /// cache the check result.
    /// </summary>
    private readonly Dictionary<HashSet<Fusion>, bool> _candidateFusionCache;

    /// <summary>
    /// Initializes a new instance of the <see cref="FusionGroupMutator"/> class.
    /// ctor.
    /// </summary>
    /// <param name="usedByAnalysisReslut">the usedby analysis.</param>
    /// <param name="fusionRule">pre order rule. </param>
    /// <param name="passOptions">pass options. </param>
    public FusionGroupMutator(IUsedByResult usedByAnalysisReslut, IMergeRewriteRule fusionRule, RunPassContext passOptions)
    {
        _usedByReslut = usedByAnalysisReslut;
        Rule = fusionRule;
        PassOptions = passOptions;
        _candidateFusionCache = new(new FusionMergeCandidateComparer());
    }

    /// <summary>
    /// Gets get the merge check cache result.
    /// </summary>
    public IReadOnlyDictionary<HashSet<Fusion>, bool> FusionMergeCandidateCache => _candidateFusionCache;

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

        if (!CompilerServices.TryMatchRoot(old_call, rule.Pattern, new() { RewriteMemo = ExpressionMemo }, out var result))
        {
            return false;
        }

        if (rule.GetReplace(
          MergedFusionRewriteCallBack, MergedFusionCheckCallBack,
          CandidateFusionCheckCallBack, CandidateFusionRecordCallBack,
          _usedByReslut, result, PassOptions) is Call replaced_call)
        {
            new_call = replaced_call;
            return true;
        }

        return false;
    }

    /// <inheritdoc/>
    public override Expr Visit(Fusion expr) => expr;

    /// <inheritdoc/>
    public override Expr MutateLeaf(Call expr)
    {
        if (TryMergeFusion(Rule, expr, out var merged_call))
        {
            return merged_call;
        }

        return expr;
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(Call expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

        if (!IsMutated)
        {
            return expr;
        }

        var with_new = expr with
        {
            Target = Visit(expr.Target),
            Parameters = MutateArray(expr.Parameters, Visit),
        };
        UpdateCallUsedBy(expr, with_new);
        return with_new;
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

    private void UpdateCallUsedBy(Call old_call, Call new_call)
    {
        /* update the usedy info */
        // 1. transfer the caller usedby info to new_call
        _usedByReslut.Transfer(old_call, new_call);

        // 2. clear all caller's and callee's usedy info
        _usedByReslut.Clear(old_call.Target, old_call);
        foreach (var param in old_call.Parameters)
        {
            _usedByReslut.Clear(param, old_call);
        }

        // 3. reset the input usedby
        _usedByReslut.Add(new_call.Target, new_call);
        foreach (var param in new_call.Parameters)
        {
            _usedByReslut.Add(param, new_call);
        }
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
