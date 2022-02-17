// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Pattern;

/// <summary>
/// Pattern for <see cref="Call"/>.
/// </summary>
/// <param name="Target">Target pattern.</param>
/// <param name="Parameters">Parameters pattern.</param>
public sealed record CallPattern(Pattern Target, VArgsPattern Parameters)
    : Pattern<Call>(x => Target.MatchLeaf(x.Target) && Parameters.MatchLeaf(x.Parameters))
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CallPattern"/> class.
    /// </summary>
    /// <param name="call"><see cref="Call"/> expression.</param>
    public CallPattern(Call call)
        : this(call.Target, new FixedVArgsPattern(call.Parameters))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CallPattern"/> class.
    /// </summary>
    /// <param name="target">Target pattern.</param>
    /// <param name="parameters">Parameter patterns.</param>
    public CallPattern(Pattern target, params ExprPattern[] parameters)
        : this(target, new FixedVArgsPattern(parameters))
    {
    }

    /// <summary>
    /// Get parameter pattern.
    /// </summary>
    /// <param name="parameter">Parameter info.</param>
    /// <returns>Parameter pattern.</returns>
    public ExprPattern this[ParameterInfo parameter]
    {
        get => Parameters[parameter.Index];
    }
}

public static partial class Utility
{
    public static CallPattern IsCall(ExprPattern Target, VArgsPattern Parameters) => new CallPattern(Target, Parameters);

    public static CallPattern IsCall(ExprPattern Target, params ExprPattern[] Parameters) => new CallPattern(Target, Parameters);
}
