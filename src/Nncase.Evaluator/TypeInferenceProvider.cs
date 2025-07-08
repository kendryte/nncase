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
    public IRType InferenceOp(Op op, ITypeInferenceContext context, Dictionary<Type, ITypeInferencer> inferencer_cache)
    {
        var op_type = op.GetType();
        if (!inferencer_cache.TryGetValue(op_type, out var inferencer))
        {
            var inferencerType = typeof(ITypeInferencer<>).MakeGenericType(op.GetType());
            inferencer = (ITypeInferencer)_serviceProvider.GetRequiredService(inferencerType);
            inferencer_cache.Add(op_type, inferencer);
        }

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
    public bool InferenceType(BaseExpr expr, Dictionary<Type, ITypeInferencer> inferencer_cache)
    {
        var visitor = new TypeInferenceVisitor(inferencer_cache);
        visitor.Visit(expr);
        return visitor.IsFullyInferenced;
    }
}
