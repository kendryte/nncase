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

namespace Nncase.IR;

/// <summary>
/// CallPrimFunction expression.
/// </summary>
public sealed class PrimFunctionWrapper : BaseFunction
{
    private static int _globalFuncIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunctionWrapper"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    /// <param name="target">Target.</param>
    /// <param name="parametersCount">Arguments count.</param>
    /// <param name="hints">the type hints.</param>
    public PrimFunctionWrapper(string name, PrimFunction target, int parametersCount, params IRType[] hints)
        : base(name, StackVMModuleKind, new Expr[] { target })
    {
        ParametersCount = parametersCount;
        TypeHints = hints;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunctionWrapper"/> class.
    /// </summary>
    /// <param name="target">Target.</param>
    /// <param name="parametersCount">Arguments count.</param>
    /// <param name="hints">the type hints.</param>
    public PrimFunctionWrapper(PrimFunction target, int parametersCount, params IRType[] hints)
        : this($"func_{_globalFuncIndex++}", target, parametersCount, hints)
    {
    }

    public PrimFunction Target => (PrimFunction)Operands[0];

    public int ParametersCount { get; }

    public IRArray<IRType> TypeHints { get; }

    /// <summary>
    /// Gets return type.
    /// </summary>
    public IRType ReturnType
    {
        get
        {
            var outputParams = Target.Parameters.AsValueEnumerable().Skip(ParametersCount).ToArray();
            return outputParams.Length == 1
                ? (TypeHints.Count <= ParametersCount ? outputParams[0].CheckedType : TypeHints[ParametersCount])
                : new TupleType(outputParams.Select((x, i) => TypeHints.Count <= ParametersCount ? x.CheckedType! : TypeHints[ParametersCount + i]));
        }
    }

    /// <inheritdoc/>
    public override IEnumerable<IRType> ParameterTypes => Target.Parameters.AsValueEnumerable().Take(ParametersCount).Select((x, i) => TypeHints.Count <= ParametersCount ? x.CheckedType : TypeHints[i]).ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitPrimFunctionWrapper(this, context);

    public PrimFunctionWrapper With(string? name = null, PrimFunction? target = null, int? parametersCount = null)
        => new PrimFunctionWrapper(name ?? Name, target ?? Target, parametersCount ?? ParametersCount);
}
