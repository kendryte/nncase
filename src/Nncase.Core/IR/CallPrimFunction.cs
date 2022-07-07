// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.TIR;

namespace Nncase.IR;

/// <summary>
/// CallPrimFunction expression.
/// </summary>
public sealed record CallPrimFunction(PrimFunction Target, IRArray<Expr> Parameters) : Expr, IParameterList<Expr>
{
    public CallAttr Attribute = CallAttr.None;

    /// <summary>
    /// Initializes a new instance of the <see cref="CallPrimFunction"/> class.
    /// </summary>
    /// <param name="target">Call target.</param>
    /// <param name="parameters">Parameters.</param>
    public CallPrimFunction(PrimFunction target, params Expr[] parameters)
        : this(target, new IRArray<Expr>(parameters.ToImmutableArray()))
    {
    }

    /// <summary>
    /// get param expr.
    /// </summary>
    /// <param name="parameter"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public Expr this[ParameterInfo parameter]
    {
        get
        {
            var type = Target.GetType();
            if (type == parameter.OwnerType)
            {
                return Parameters[parameter.Index];
            }
            else
            {
                throw new ArgumentOutOfRangeException($"Target {Target} doesn't have parameter: {parameter.Name}.");
            }
        }
    }
}
