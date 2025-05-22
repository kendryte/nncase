// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.IR;

/// <summary>
/// CallFunction expression.
/// </summary>
public sealed class FunctionWrapper : BaseFunction
{
    private static int _globalFuncIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="FunctionWrapper"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    /// <param name="moduleKind">Module kind.</param>
    /// <param name="target">Target.</param>
    public FunctionWrapper(string name, string moduleKind, BaseFunction target)
        : base(name, moduleKind, [target])
    {
        if (target is not Function or PrimFunctionWrapper)
        {
            throw new ArgumentException($"target must be {nameof(Function)} or {nameof(PrimFunctionWrapper)}");
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="FunctionWrapper"/> class.
    /// </summary>
    /// <param name="target">Target.</param>
    /// <param name="moduleKind">Module kind.</param>
    public FunctionWrapper(string moduleKind, BaseFunction target)
        : this($"functionwrapper_{_globalFuncIndex++}", moduleKind, target)
    {
    }

    public BaseFunction Target => (BaseFunction)Operands[0];

    /// <inheritdoc/>
    public override IEnumerable<IRType> ParameterTypes => Target.ParameterTypes;

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitFunctionWrapper(this, context);

    public FunctionWrapper With(string? name = null, string? moduleKind = null, BaseFunction? target = null)
        => new FunctionWrapper(name ?? Name, moduleKind ?? ModuleKind, target ?? Target);
}
