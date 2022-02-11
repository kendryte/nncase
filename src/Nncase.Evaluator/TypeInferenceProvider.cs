// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;

namespace Nncase.Evaluator;

internal class TypeInferenceProvider : ITypeInferenceProvider
{
    private readonly IServiceProvider _serviceProvider;

    public TypeInferenceProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    /// <inheritdoc/>
    public IRType InferenceOp(Op op, ITypeInferenceContext context)
    {
        // TODO: Add inferencers cache.
        var inferencerType = typeof(ITypeInferencer<>).MakeGenericType(op.GetType());
        var inferencer = (ITypeInferencer)_serviceProvider.GetRequiredService(inferencerType);

        try
        {
            return inferencer.Visit(context, op);
        }
        catch (TypeInferenceInterruptException ex)
        {
            return ex.ReasonType;
        }
    }

    /// <inheritdoc/>
    public bool InferenceType(Expr expr)
    {
        var visitor = new TypeInferenceVisitor();
        visitor.Visit(expr);
        return visitor.IsFullyInferenced;
    }
}
