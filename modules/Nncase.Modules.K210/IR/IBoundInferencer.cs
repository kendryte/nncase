namespace Nncase.IR;

/// <summary>
/// output_bounds inferencer interface.
/// </summary>
public interface IBoundInferencer
{
    /// <summary>
    /// Inference op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <param name="output_bounds">given output bounds.</param>
    /// <returns>Result.</returns>
    void Visit(IBridgeBoundsInferContext context, Op target, IRArray<TIR.Range> output_bounds);


    static void CheckCallBounds(IBridgeBoundsInferContext context, IRArray<TIR.Range> output_bounds)
    {
        var output_shape = context.CurrentCallShape;
        if (output_shape.Count != output_bounds.Count)
            throw new ArgumentOutOfRangeException($"The Give Bounds {output_bounds.Count} Is Not In Shape {output_shape.Count}");
    }

    /// <summary>
    /// visit tile step
    /// </summary>
    /// <param name="ctx"></param>
    /// <param name="target"></param>
    void VisitTileStep(IBridgeBoundsInferContext ctx, Op target);
}

/// <summary>
/// output_bounds inferencer interface.
/// </summary>
public interface IBoundInferencer<T> : IBoundInferencer
    where T : Op
{
    /// <summary>
    /// Inference bound of op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <param name="output_bounds">given output shape.</param>
    /// <returns>Result.</returns>
    void Visit(IBridgeBoundsInferContext context, T target, IRArray<TIR.Range> output_bounds);

    void IBoundInferencer.Visit(IBridgeBoundsInferContext ctx, Op target, IRArray<TIR.Range> output_bounds)
    {
        IBoundInferencer.CheckCallBounds(ctx, output_bounds);
        Visit(ctx, (T)target, output_bounds);
    }

    void VisitTileStep(IBridgeBoundsInferContext ctx, T target);

    void IBoundInferencer.VisitTileStep(IBridgeBoundsInferContext ctx, Op target)
    {
        VisitTileStep(ctx, (T)target);
    }
}
