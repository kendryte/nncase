using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Evaluator;

/// <summary>
/// this attribute mark the source generator auto generate IEvaluator's interface impl
/// </summary>
public class EvaluatorGeneratorAttribute : Attribute
{
}

/// <summary>
/// Evaluator interface.
/// </summary>
public interface IEvaluator
{
    /// <summary>
    /// Evaluate op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    IValue Visit(IEvaluateContext context, Op target);
}

/// <summary>
/// Evaluator interface.
/// </summary>
public interface IEvaluator<T> : IEvaluator
    where T : Op
{
    /// <summary>
    /// Evaluate op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    IValue Visit(IEvaluateContext context, T target);

    IValue IEvaluator.Visit(IEvaluateContext context, Op target)
    {
        return Visit(context, (T)target);
    }
}
