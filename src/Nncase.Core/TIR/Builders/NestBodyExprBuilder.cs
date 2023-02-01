// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR.Builders;

/// <summary>
/// GridWrapper for collect the for item.
/// </summary>
internal class NestBodyExprBuilder<T> : ISequentialBuilder<T>
    where T : Expr
{
    private readonly ISequentialBuilder<T>[] _subBuilders;

    /// <summary>
    /// Initializes a new instance of the <see cref="NestBodyExprBuilder{T}"/> class.
    /// ctor.
    /// <remarks>
    /// NOTE We will auto add exprs to nest list!
    /// </remarks>
    /// </summary>
    public NestBodyExprBuilder(params ISequentialBuilder<T>[] subBuilders)
    {
        _subBuilders = subBuilders;
    }

    /// <summary>
    /// Wrapper Body method.
    /// </summary>
    public ISequentialBuilder<T> Body(params object[] exprOrBuilders)
    {
        _subBuilders[_subBuilders.Length - 1].Body(exprOrBuilders);
        return this;
    }

    public T Build()
    {
        for (int i = _subBuilders.Length - 2; i >= 0; i--)
        {
            _subBuilders[i].Body(_subBuilders[i + 1].Build());
        }

        return _subBuilders[0].Build();
    }

    public ISequentialBuilder<T> InsertBody(int index, params object[] exprOrBuilders)
    {
        _subBuilders[_subBuilders.Length - 1].InsertBody(index, exprOrBuilders);
        return this;
    }
}
