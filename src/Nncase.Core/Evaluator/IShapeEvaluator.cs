using Nncase.IR;

namespace Nncase.Evaluator;


/// <summary>
/// Shape Evaluator interface.
/// </summary>
public interface IShapeEvaluator
{
    /// <summary>
    /// Evaluate op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    Expr Visit(IShapeEvaluateContext context, Op target);
}

/// <summary>
/// Shape Evaluator interface.
/// </summary>
public interface IShapeEvaluator<T> : IShapeEvaluator
    where T : Op
{
    /// <summary>
    /// Evaluate shape of op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    Expr Visit(IShapeEvaluateContext context, T target);

    Expr IShapeEvaluator.Visit(IShapeEvaluateContext ctx, Op target)
    {
        return Visit(ctx, (T)target);
    }
}

/// <summary>
/// this attribute mark the source generator auto generate IShapeEvaluator's interface impl.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
public sealed class ShapeEvaluatorGeneratorAttribute : Attribute
{
}
