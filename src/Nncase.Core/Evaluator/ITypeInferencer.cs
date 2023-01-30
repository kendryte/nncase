// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Evaluator;

/// <summary>
/// Type inferencer interface.
/// </summary>
public interface ITypeInferencer
{
    /// <summary>
    /// Inference op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    IRType Visit(ITypeInferenceContext context, Op target);
}

/// <summary>
/// Type inferencer interface.
/// </summary>
public interface ITypeInferencer<T> : ITypeInferencer
    where T : Op
{
    /// <summary>
    /// Inference type of op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    IRType Visit(ITypeInferenceContext context, T target);

    IRType ITypeInferencer.Visit(ITypeInferenceContext ctx, Op target)
    {
        return Visit(ctx, (T)target);
    }
}

/// <summary>
/// this attribute mark the source generator auto generate ITypeInferencer's interface impl.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
public sealed class TypeInferGeneratorAttribute : Attribute
{
}
