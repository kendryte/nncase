// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR.Builders;

/// <summary>
/// Build the sequential.
/// </summary>
/// <typeparam name="T">Result type.</typeparam>
public interface ISequentialBuilder<out T> : IExprBuilder<T>
    where T : Expr
{
    /// <summary>
    /// Add the expr items to body.
    /// </summary>
    /// <param name="exprOrBuilders">Expressions.</param>
    /// <returns>Result.</returns>
    ISequentialBuilder<T> Body(params object[] exprOrBuilders);

    /// <summary>
    /// Insert the expr items to body.
    /// </summary>
    ISequentialBuilder<T> InsertBody(int index, params object[] exprOrBuilders);
}

internal class SequentialBuilder<T> : ISequentialBuilder<T>
    where T : Expr
{
    private readonly Func<Sequential, T> _creator;
    private readonly List<object> _body = new();

    public SequentialBuilder(Func<Sequential, T> creator)
    {
        _creator = creator;
    }

    public ISequentialBuilder<T> Body(params object[] exprOrBuilders)
    {
        _body.AddRange(exprOrBuilders);
        return this;
    }

    public T Build()
    {
        return _creator(Sequential.Flatten(CollectionsMarshal.AsSpan(_body)));
    }

    public ISequentialBuilder<T> InsertBody(int index, params object[] exprOrBuilders)
    {
        _body.InsertRange(index, exprOrBuilders);
        return this;
    }
}
