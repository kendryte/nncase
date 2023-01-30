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
/// <param name="Name">name.</param>
public sealed record VArgsPattern(Func<IReadOnlyList<Expr>, IRArray<Pattern>> FieldsGenerator, string? Name)
    : Pattern(Name), IPattern<IReadOnlyList<Expr>>, IReadOnlyList<Pattern>
{
    private IRArray<Pattern> _fields;

    /// <summary>
    /// Initializes a new instance of the <see cref="VArgsPattern"/> class.
    /// </summary>
    /// <param name="fields">Fields patterns.</param>
    /// <param name="name">name.</param>
    public VArgsPattern(IRArray<Pattern> fields, string? name)
        : this(x => fields, name)
    {
        _fields = fields;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="VArgsPattern"/> class.
    /// </summary>
    /// <param name="fields">Fields patterns.</param>
    /// <param name="name">name.</param>
    public VArgsPattern(IRArray<Expr> fields, string? name)
        : this(x => fields.Select(f => (Pattern)f).ToArray(), name)
    {
        _fields = fields.Select(f => (Pattern)f).ToArray();
    }

    /// <inheritdoc/>
    public int Count => _fields.Count;

    /// <summary>
    /// Gets a value indicating whether check the fields is empty.
    /// </summary>
    public bool IsDefaultOrEmpty => _fields.IsDefaultOrEmpty;

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
    public override bool MatchLeaf(object input) => input is IReadOnlyList<Expr> exprs and not Expr && MatchLeaf(exprs);
}

public partial class Utility
{
    /// <summary>
    /// create the VArgsPattern match give the patterns.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="parameters">params.</param>
    /// <returns>VArgsPattern .</returns>
    public static VArgsPattern IsVArgs(string? name, Pattern[] parameters) => new VArgsPattern(parameters, name);

    /// <summary>
    /// create VargsPattern without name.
    /// </summary>
    /// <param name="parameters">parameters.</param>
    /// <returns>VArgsPattern.</returns>
    public static VArgsPattern IsVArgs(params Pattern[] parameters)
      => IsVArgs(null, parameters);

    /// <summary>
    /// Create repeated Vargs by template pattern, eg. give the const pattern as Template, will match {Const(),...Const()}.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="creator">dynamic creator for generator ExprPattern as template.</param>
    /// <returns>VArgsPattern.</returns>
    public static VArgsPattern IsVArgsRepeat(string? name, Func<Pattern> creator) => IsVArgsRepeat(
        name, input =>
        {
            var patterns = new Pattern[input.Count];
            for (int i = 0; i < patterns.Length; i++)
            {
                patterns[i] = creator();
            }

            return patterns;
        });

    /// <summary>
    /// <see cref="IsVArgsRepeat(string?, Func{Pattern})"/>.
    /// </summary>
    public static VArgsPattern IsVArgsRepeat(Func<Pattern> creator) => IsVArgsRepeat(null, creator);

    /// <summary>
    /// Create repeated Vargs match pattern, it will manual clear the inner container.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="creator">the int mean matched params nums, list[pattern] is inner params contianer. </param>
    /// <returns>VArgsPattern.</returns>
    public static VArgsPattern IsVArgsRepeat(string? name, Func<IReadOnlyList<Expr>, IRArray<Pattern>> creator)
      => new VArgsPattern(creator, name);

    public static VArgsPattern IsVArgsRepeat(Func<IReadOnlyList<Expr>, IRArray<Pattern>> creator) => IsVArgsRepeat(null, creator);
}
