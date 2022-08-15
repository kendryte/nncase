namespace Nncase.IR;


/// <summary>
/// the segment
/// </summary>
/// <param name="Start"></param>
/// <param name="Stop"></param>
/// <param name="Step"></param>
public record Segment(int Start, int Stop, int Step)
{
    /// <summary>
    /// get length.
    /// </summary>
    public int Length => Stop - Start;

    /// <summary>
    /// All Range
    /// </summary>
    public readonly static Segment All = new(int.MinValue, int.MaxValue, 1);

    public static implicit operator Segment(System.Range range)
    {
        if (range.Equals(System.Range.All))
            return Segment.All;
        if (range.Start.IsFromEnd)
            throw new NotSupportedException();
        if (range.End.IsFromEnd)
            throw new NotSupportedException();
        return new(range.Start.Value, range.End.Value, 1);
    }

    public override string ToString()
    {
        return $"({Start},{Stop},{Step})";
    }
}

public interface IBoundsInferGraph
{
    /// <summary>
    /// Get the Bounds Function
    /// </summary>
    /// <param name="expr">Expr expr.</param>
    /// <returns></returns>
    public IBoundsFunction this[Expr expr] { get; }

    /// <summary>
    /// Get the expressions by cache level.
    /// </summary>
    /// <param name="cache_level"> 1,2,3 ..</param>
    /// <returns>Exprs.</returns>
    public IEnumerable<Expr> GetCachedExprs(int cache_level);

    /// <summary>
    /// Get the Bounds inference Function Expression.
    /// </summary>
    /// <param name="expr"></param>
    /// <returns></returns>
    public Function GetFunction(Expr expr);

    /// <summary>
    /// Set the root Expr's segments.
    /// </summary>
    public List<Segment> RootSegment { get; set; }

    /// <summary>
    /// Get the root Expr's tile search steps.
    /// </summary>
    public ReadOnlySpan<Segment> RootTileStep { get; }

    /// <summary>
    /// Set the root Expr's Bounds Exprssion
    /// </summary>
    public List<TIR.Range> RootBounds { get; set; }

    /// <summary>
    /// get the env
    /// </summary>
    //public Transform.Rules.K510.GNNEEnv Env { get; }
}

/// <summary>
/// the bounds function.
/// </summary>
public interface IBoundsFunction
{

    /// <summary>
    /// the full bounds infer function.
    /// </summary>
    public Function FullBoundsFunction { get; init; }

    /// <summary>
    /// the root Bounds. 
    /// </summary>
    public IReadOnlyList<TIR.Range> RootBounds { get; }

    /// <summary>
    /// the root Segment.
    /// </summary>
    public IReadOnlyList<Segment> RootSegment { get; }

    /// <summary>
    /// 给定一个root的bounds 推理出当前的bounds
    /// </summary>
    /// <param name="cur_root_bounds"></param>
    /// <returns></returns>
    public IReadOnlyList<TIR.Range> Infer(IEnumerable<TIR.Range> cur_root_bounds);

    /// <summary>
    /// give a root segment, return the current eval function's segments
    /// </summary>
    /// <param name="cur_root_segments"></param>
    /// <returns></returns>
    public Segment[] Eval(params Segment[] cur_root_segments)
    {
        var call = new Call(FullBoundsFunction, cur_root_segments.Select(rg => new Expr[] { rg.Start, rg.Stop, rg.Step }).SelectMany(rg => rg).ToArray()).Evaluate();
        return call.AsTensors().Chunk(3).Select(arr =>
          new Segment(arr[0].ToScalar<int>(), arr[1].ToScalar<int>(), arr[2].ToScalar<int>())).ToArray();
    }

    /// <summary>
    /// 给定一个no pad 的sub bounds 反推出当前的bounds
    /// </summary>
    /// <param name="sub_no_pad_bounds"></param>
    /// <returns> the sub bounds. </returns>
    public IEnumerable<TIR.Range> BackWardBounds(IEnumerable<TIR.Range> sub_no_pad_bounds);

    /// <summary>
    /// get the mutiple dim's index bounds;
    /// </summary>
    /// <returns></returns>
    public Segment[] Segments { get; }

    /// <summary>
    /// get the tile shape
    /// </summary>
    public IR.Shape TileShape { get; }

    /// <summary>
    /// get the expr's shape
    /// </summary>
    public IR.Shape Shape { get; }

    /// <summary>
    /// get the padding expressions.
    /// </summary>
    public IReadOnlyList<(Expr Before, Expr After)> Paddings { get; }

    /// <summary>
    /// Get the current bounds.
    /// </summary>
    public IReadOnlyList<TIR.Range> Bounds { get; }

    /// <summary>
    /// Get the current bounds and remove the padding.
    /// <example>
    /// when the bounds is [-2,4)
    /// the no pad bounds  is [0,4)
    /// </example>
    /// </summary>
    public IReadOnlyList<TIR.Range> NoPadBounds { get; }

    /// <summary>
    /// Get the Bounds which clamped by shape.
    /// </summary>
    public IReadOnlyList<TIR.Range> ClampedBounds { get; }

}