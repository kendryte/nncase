// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for varadic expressions.
/// </summary>
/// <param name="FieldsGenerator">Fields patterns generator.</param>
public sealed record VArgsPattern(Func<IReadOnlyList<Expr>, IRArray<Pattern>> FieldsGenerator)
    : Pattern, IPattern<IReadOnlyList<Expr>>, IReadOnlyList<Pattern>
{
    private IRArray<Pattern> _fields;

    /// <summary>
    /// Initializes a new instance of the <see cref="VArgsPattern"/> class.
    /// </summary>
    /// <param name="fields">Fields patterns.</param>
    public VArgsPattern(IRArray<Pattern> fields)
        : this(x => fields)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="VArgsPattern"/> class.
    /// </summary>
    /// <param name="fields">Fields patterns.</param>
    public VArgsPattern(IRArray<Expr> fields)
        : this(x => fields.Select(f => (Pattern)f).ToArray())
    {
    }

    /// <inheritdoc/>
    public int Count => _fields.Count;

    /// <inheritdoc/>
    public Pattern this[int index] => _fields[index];

    /// <inheritdoc/>
    public IEnumerator<Pattern> GetEnumerator() => _fields.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <inheritdoc/>
    public bool MatchLeaf(IReadOnlyList<Expr> input)
    {
        _fields = FieldsGenerator(input);

        if (input.Count != _fields.Count)
        {
            return false;
        }

        return true;
    }

    /// <inheritdoc/>
    public override bool MatchLeaf(object input) => input is IReadOnlyList<Expr> exprs && MatchLeaf(exprs);
}

public partial class Utility
{
    public static VArgsPattern IsVArgs(params ExprPattern[] Parameters)
      => new VArgsPattern(Parameters);

    /// <summary>
    /// Create repeated Vargs by template pattern, eg. give the const pattern as Template, will match {Const(),...Const()}.
    /// </summary>
    /// <param name="creator">dynamic creator for generator ExprPattern as template.</param>
    /// <returns>VArgsPattern.</returns>
    public static VArgsPattern IsVArgsRepeat(Func<Pattern> creator) => IsVArgsRepeat((input) =>
    {
        var patterns = new Pattern[input.Count];
        for (int i = 0; i < patterns.Length; i++)
        {
            patterns[i] = creator();
        }

        return patterns;
    });

    /// <summary>
    /// Create repeated Vargs match pattern, it will manual clear the inner container.
    /// </summary>
    /// <param name="creator">the int mean matched params nums, list[pattern] is inner params contianer. </param>
    /// <returns>VArgsPattern.</returns>
    public static VArgsPattern IsVArgsRepeat(Func<IReadOnlyList<Expr>, IRArray<Pattern>> creator)
      => new VArgsPattern(creator);
}
