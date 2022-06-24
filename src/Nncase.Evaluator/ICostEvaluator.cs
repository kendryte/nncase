// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;

namespace Nncase.Evaluator;

/// <summary>
/// this attribute mark the source generator auto generate ICostEvaluator's interface impl
/// </summary>
public class CostEvaluatorGeneratorAttribute : Attribute
{

}

/// <summary>
/// Cost evaluator interface.
/// </summary>
public interface ICostEvaluator
{
    /// <summary>
    /// Evaluate op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    Cost? Visit(ICostEvaluateContext context, Op target);
}

/// <summary>
/// Cost evaluator interface.
/// </summary>
public interface ICostEvaluator<T> : ICostEvaluator
    where T : Op
{
    /// <summary>
    /// Evaluate op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    Cost? Visit(ICostEvaluateContext context, T target);

    Cost? ICostEvaluator.Visit(ICostEvaluateContext context, Op target)
    {
        return Visit(context, (T)target);
    }
}
