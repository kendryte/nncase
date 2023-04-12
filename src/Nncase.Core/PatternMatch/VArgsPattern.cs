// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.PatternMatch;

public delegate IReadOnlyList<Pattern> PatternGenerator(ReadOnlySpan<Expr> exprs);

/// <summary>
/// Pattern for varadic expressions.
/// </summary>
/// <param name="FieldsGenerator">Fields patterns generator.</param>
/// <param name="Name">name.</param>
public sealed record VArgsPattern(PatternGenerator FieldsGenerator, string? Name)
    : Pattern(Name), IReadOnlyList<Pattern>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="VArgsPattern"/> class.
    /// </summary>
    /// <param name="fields">Fields patterns.</param>
    /// <param name="name">name.</param>
    public VArgsPattern(IReadOnlyList<Pattern> fields, string? name)
        : this(x => fields, name)
    {
        Fields = fields;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="VArgsPattern"/> class.
    /// </summary>
    /// <param name="fields">Fields patterns.</param>
    /// <param name="name">name.</param>
    public VArgsPattern(IEnumerable<Expr> fields, string? name)
        : this(x => fields.Select(f => (Pattern)f).ToArray(), name)
    {
        Fields = fields.Select(f => (Pattern)f).ToArray();
    }

    public IReadOnlyList<Pattern> Fields { get; private set; } = Array.Empty<Pattern>();

    public int Count => Fields.Count;

    public Pattern this[int index] => Fields[index];

    public IEnumerator<Pattern> GetEnumerator() => Fields.GetEnumerator();

    public bool MatchLeaf(ReadOnlySpan<Expr> input)
    {
        Fields = FieldsGenerator(input);

        if (input.Length != Fields.Count)
        {
            return false;
        }

        return true;
    }

    /// <inheritdoc/>
    public override bool MatchLeaf(Expr input) => false;

    IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)Fields).GetEnumerator();
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
            var patterns = new Pattern[input.Length];
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
    /// <param name="generator">the int mean matched params nums, list[pattern] is inner params contianer. </param>
    /// <returns>VArgsPattern.</returns>
    public static VArgsPattern IsVArgsRepeat(string? name, PatternGenerator generator)
      => new VArgsPattern(generator, name);

    public static VArgsPattern IsVArgsRepeat(PatternGenerator generator) => IsVArgsRepeat(null, generator);
}
