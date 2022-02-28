// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// the interface that we can use parameterinfo the parameter.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IParameterList<T>
    {
        /// <summary>
        /// get parameter info
        /// </summary>
        /// <param name="parameter"></param>
        /// <returns></returns>
        public T this[ParameterInfo parameter] { get; }
    }

    /// <summary>
    /// Call expression.
    /// </summary>
    public sealed record Call(Expr Target, IRArray<Expr> Parameters) : Expr, IParameterList<Expr>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Call"/> class.
        /// </summary>
        /// <param name="target">Call target.</param>
        /// <param name="parameters">Parameters.</param>
        public Call(Expr target, params Expr[] parameters)
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
}
