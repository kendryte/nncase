
using Microsoft.Extensions.DependencyInjection;
using Nncase.TIR;
using Nncase.Transform.Rules.K210;

namespace Nncase.IR;


public interface IBoundInferencerProvider
{
    /// <summary>
    /// inference op input's segment.
    /// </summary>
    /// <param name="ctx"></param>
    /// <param name="target"></param>
    /// <param name="bounds"></param>
    public void BridgeBoundsInferOp(IBridgeBoundsInferContext ctx, Op target, IRArray<TIR.Range> bounds);

    /// <summary>
    /// inference the all segment expression from the root node.
    /// </summary>
    /// <param name="root">the root expression.</param>
    /// <param name="leaf">the leaf expression.</param>
    /// <returns></returns>
    public IBoundsInferGraph MakeBoundsInferGraph(Expr root, Expr? leaf);

    /// <summary>
    /// inference op output tile step.
    /// </summary>
    /// <param name="ctx"></param>
    /// <param name="target"></param>
    void BridgeTileStepOp(IBridgeBoundsInferContext ctx, Op target);
}


sealed internal class BoundsFunction : IBoundsFunction
{
    /// <inheritdoc/>
    public Function FullBoundsFunction { get; init; }

    /// <inheritdoc/>
    public IReadOnlyList<Segment> RootSegment => _inferGraph.RootSegment;

    /// <inheritdoc/>
    public IReadOnlyList<TIR.Range> RootBounds => _inferGraph.RootBounds;

    /// <inheritdoc/>
    public Segment[] Segments => ((IBoundsFunction)this).Eval(RootSegment.ToArray());

    /// <inheritdoc/>
    public Shape Shape { get; init; }

    /// <inheritdoc/>
    public Shape TileShape => new(Segments.ToArray().Select(s => s.Length));

    /// <inheritdoc/>
    public IReadOnlyList<(Expr Before, Expr After)> Paddings
    {
        get
        {
            if (!_padding.Any())
                _padding.AddRange(TIRUtilities.ComputePaddings(Bounds, Shape));
            return _padding.AsReadOnly();
        }
    }

    public BoundsFunction(IBoundsInferGraph inferGraph, Function func, Shape shape)
    {
        _inferGraph = inferGraph;
        FullBoundsFunction = func;
        Shape = shape;
    }

    /// <inheritdoc/>
    public IReadOnlyList<TIR.Range> Bounds
    {
        get
        {
            if (RootBounds.Count == 0)
                throw new InvalidOperationException("Please Set RootBounds!");

            if (!_current_bounds.Any())
            {
                var call = new Call(FullBoundsFunction, RootBounds.Select(rg => new Expr[] { rg.Start, rg.Stop, rg.Step }).SelectMany(rg => rg).ToArray());
                var ranges = ((Tuple)FullBoundsFunction.Body).Select((_, i) => i).Chunk(3).Select(arr =>
                     new TIR.Range(call[arr[0]], call[arr[1]], call[arr[2]])
                 ).ToArray();
                _current_bounds.AddRange(ranges);
            }
            return _current_bounds.AsReadOnly();
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<TIR.Range> Infer(IEnumerable<TIR.Range> cur_root_bounds)
    {
        if ((cur_root_bounds.Count() * 3) != FullBoundsFunction.Parameters.Count)
            throw new InvalidOperationException("The cur_root_bounds number != FullBoundsFunction parameter.");

        var call = new Call(FullBoundsFunction, cur_root_bounds.Select(rg => new Expr[] { rg.Start, rg.Stop, rg.Step }).SelectMany(rg => rg).ToArray());

        return ((Tuple)FullBoundsFunction.Body).Select((_, i) => i).Chunk(3).Select(arr =>
             new TIR.Range(call[arr[0]], call[arr[1]], call[arr[2]])
         ).ToList().AsReadOnly();
    }

    /// <inheritdoc/>
    public IEnumerable<TIR.Range> BackWardBounds(IEnumerable<TIR.Range> sub_no_pad_bounds) =>
      TIRUtilities.ComputeBounds(sub_no_pad_bounds, Bounds, Paddings);


    /// <inheritdoc/>
    public IReadOnlyList<TIR.Range> NoPadBounds
    {
        get
        {
            if (!_validBounds.Any())
            {
                _validBounds.AddRange(TIRUtilities.ComputeNoPadBounds(Bounds, Paddings));
            }
            return _validBounds.AsReadOnly();
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<TIR.Range> ClampedBounds
    {
        get
        {
            if (!_clampedBounds.Any())
            {
                _clampedBounds.AddRange(TIRUtilities.ClampBounds(Bounds, Shape));
            }
            return _clampedBounds.AsReadOnly();
        }
    }

    private readonly List<TIR.Range> _current_bounds = new();
    private readonly List<TIR.Range> _validBounds = new();
    private readonly List<TIR.Range> _clampedBounds = new();

    private readonly List<(Expr Before, Expr After)> _padding = new();
    private IBoundsInferGraph _inferGraph;
}

sealed internal class BridgeBoundsInferContext : IBridgeBoundsInferContext
{
    /// <summary>
    /// only contain the valid buffer expr 
    /// </summary>
    public readonly Dictionary<Expr, IRArray<TIR.Range>> ExprMemo = new(ReferenceEqualityComparer.Instance);

    public readonly Dictionary<Expr, IR.Segment[]> TileStepMemo = new(ReferenceEqualityComparer.Instance);

    /// <summary>
    /// 收集每个expr对应的cache位置.
    /// </summary>
    public readonly Dictionary<int, HashSet<Expr>> CachedExpr = new();

    public Call? _currentCall;

    public Call CurrentCall
    {
        get
        {
            if (_currentCall is null)
                throw new InvalidOperationException("Must Be set first!");
            return _currentCall;
        }
        set
        {
            if (_currentCall is null)
                ExprMemo[value] = Root;
            _currentCall = value;
        }
    }

    IRArray<TIR.Range> Root;

    public BridgeBoundsInferContext(IRArray<TIR.Range> root)
    {
        Root = root;
        //Env = new(true, false);
    }

    public Expr GetArgument(Op op, ParameterInfo parameter) => CurrentCall[parameter];

    public Shape GetArgumentShape(Op op, ParameterInfo parameter) => GetArgument(op, parameter).CheckedShape;

    public Tensor GetArgumentTensor(Op op, ParameterInfo parameter) => GetArgument(op, parameter) switch
    {
        TensorConst @const => @const.Value,
        var x => throw new ArgumentOutOfRangeException(x.GetType().Name),
    };

    public Shape CurrentCallShape => CurrentCall.CheckedShape;

    public void SetArgumentBounds(Op op, ParameterInfo parameter, IRArray<TIR.Range> segments, int cache_level)
    {
        if (!CachedExpr.TryGetValue(cache_level, out var set))
        {
            set = new(ReferenceEqualityComparer.Instance);
            CachedExpr.Add(cache_level, set);
        }
        var key = GetArgument(op, parameter);
        if (key is None)
            return;
        set.Add(key);
        ExprMemo.Add(key, segments);
    }

    public ReadOnlySpan<IR.Segment> GetArgumentTileStep(Op op, ParameterInfo parameter) => TileStepMemo[CurrentCall[parameter]];

    public void SetTileStep(ReadOnlySpan<IR.Segment> tile_step)
    {
        if (!TileStepMemo.TryGetValue(CurrentCall, out var cur_steps))
        {
            cur_steps = new IR.Segment[tile_step.Length];
            tile_step.CopyTo(cur_steps);
        }
        else
        {
            var new_steps = cur_steps.Zip(tile_step.ToArray()).
                Select(t =>
                {
                    if (t.Item1.Start != t.Item2.Start)
                        throw new InvalidDataException();
                    if (t.Item1.Stop != t.Item2.Stop)
                        throw new InvalidDataException();
                    return t.Item1 with
                    {
                        Step = ((IBridgeBoundsInferContext)this).GetMinimumCommonMultiple(t.Item1.Step, t.Item2.Step)
                    };
                }).ToArray();
            new_steps.CopyTo(cur_steps, 0);
        }

        TileStepMemo.Add(CurrentCall, cur_steps);
    }

    public IRArray<TIR.Range> CallBounds => ExprMemo[CurrentCall];

    //public Transform.Rules.K510.GNNEEnv Env { get; init; }
}

/// <summary>
/// visitor for build the bounds infer graph.
/// </summary>
sealed internal class BoundsInferGraphVisitor : ExprVisitor<bool, bool>
{
    public readonly BridgeBoundsInferContext Context;
    private readonly Expr? _leaf;

    public BoundsInferGraphVisitor(IRArray<TIR.Range> root, Expr? leaf)
    {
        Context = new(root);
        _leaf = leaf;
    }

    /// <inheritdoc/>
    public override bool DefaultVisitLeaf(Expr expr)
    {
        return true;
    }

    /// <inheritdoc/>
    public override bool VisitLeaf(Var expr)
    {
        var step = expr.CheckedShape.Select(dim => new IR.Segment(1, dim.FixedValue, 1)).ToArray();
        Context.TileStepMemo.Add(expr, step);
        return true;
    }

    public override bool VisitLeaf(Const expr)
    {
        if (expr is TensorConst tc)
        {
            var step = expr.CheckedShape.Select(dim => new IR.Segment(1, dim.FixedValue, 1)).ToArray();
            Context.TileStepMemo.Add(tc, step);
        }
        else
        {
            throw new NotSupportedException(expr.GetType().ToString());
        }
        return true;
    }

    public override bool VisitLeaf(Call expr)
    {
        Context.CurrentCall = expr;
        switch (expr.Target)
        {
            case Op op:
                ExtCompilerServices.BridgeTileStepOp(Context, op);
                break;
            default:
                throw new NotSupportedException("Only Can Dealwith OP");
        };
        return true;
    }

    /// <inheritdoc/>
    public override bool Visit(Call expr)
    {
        Context.CurrentCall = expr;
        switch (expr.Target)
        {
            case Op op:
                ExtCompilerServices.BridgeBoundsInferOp(Context, op, Context.CallBounds);
                break;
            default:
                throw new NotSupportedException("Only Can Dealwith OP");
        };
        return base.Visit(expr);
    }

}

sealed class BoundsInferGraph : IBoundsInferGraph
{
    public readonly BridgeBoundsInferContext Context;

    private readonly Var[] _funcParamVars;
    private Expr _root_expr;
    private readonly List<Segment> _rootSegment = new();
    private readonly List<TIR.Range> _rootBounds = new();
    private readonly Dictionary<Expr, IBoundsFunction> _boundsFuncCache = new(ReferenceEqualityComparer.Instance);

    /// <inheritdoc/>
    public List<Segment> RootSegment
    {
        get
        {
            return _rootSegment;
        }
        set
        {
            if (_rootSegment.Count != 0)
            {
                throw new InvalidOperationException("Can't Update Twice!");
            }
            _rootSegment.AddRange(value);
        }
    }

    /// <inheritdoc/>
    public List<TIR.Range> RootBounds
    {
        get
        {
            return _rootBounds;
        }
        set
        {
            if (_rootBounds.Count != 0)
            {
                throw new InvalidOperationException("Can't Update Twice!");
            }
            _rootBounds.AddRange(value);
        }
    }

    //public GNNEEnv Env { get; }

    public ReadOnlySpan<Segment> RootTileStep => Context.TileStepMemo[_root_expr];

    public BoundsInferGraph(BridgeBoundsInferContext context, Expr root_expr, (Var, Var, Var)[] segment_vars)
    {
        Context = context;
        _funcParamVars = segment_vars.Select(t => new[] { t.Item1, t.Item2, t.Item3 }).SelectMany(arr => arr).ToArray();
        _root_expr = root_expr;
    }

    /// <inheritdoc/>
    public Function GetFunction(Expr expr)
    {
        // NOTE 返回值是 expr的dims * 3 的tuple, 输入是output dims * 3. 
        var func = new Function(
            new IR.Tuple(Context.ExprMemo[expr].
                Select(rg => new[] { rg.Start, rg.Stop, rg.Step }).
                SelectMany(arr => arr).
                ToArray()),
            _funcParamVars);
        return func;
    }

    /// <inheritdoc/>
    public IBoundsFunction this[Expr expr]
    {
        get
        {
            if (!_boundsFuncCache.TryGetValue(expr, out var func))
            {
                func = new BoundsFunction(this, GetFunction(expr), expr.CheckedShape);
                _boundsFuncCache[expr] = func;
            }
            return func;
        }
    }

    /// <inheritdoc/>
    public IEnumerable<Expr> GetCachedExprs(int cache_level) => Context.CachedExpr[cache_level];

}

sealed internal class BoundInferencerProvider : IBoundInferencerProvider
{

    private readonly IServiceProvider _serviceProvider;

    public BoundInferencerProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    /// <inheritdoc/>
    public IBoundsInferGraph MakeBoundsInferGraph(Expr root, Expr? leaf)
    {
        // 构造出segment的输入, 是三个var.
        var root_segments_var = root.CheckedShape.Select(d =>
        {
            var start = new Var(TensorType.Scalar(DataTypes.Int32));
            var end = new Var(TensorType.Scalar(DataTypes.Int32));
            var step = new Var(TensorType.Scalar(DataTypes.Int32));
            return (start, end, step);
        }).ToArray();

        // 用两个require限制输入的segment必须是位于正确范围中的.
        var visitor = new BoundsInferGraphVisitor(
          root.CheckedShape.Zip(root_segments_var).Select((tp, i) =>
          new TIR.Range(
            IR.F.Math.Require(tp.Item2.start >= 0, tp.Item2.start, $"Dim {i} Start >= 0"),
            IR.F.Math.Require(tp.Item2.end <= tp.Item1.FixedValue, tp.Item2.end, $"Dim {i} End <= {tp.Item1.FixedValue}"),
            IR.F.Math.Require(IR.F.Math.Equal(tp.Item2.step, 1), 1, $"Dim {i} Step == 1"))).ToArray(),
            leaf);
        visitor.Visit(root);

        return new BoundsInferGraph(visitor.Context, root, root_segments_var);
    }

    /// <inheritdoc/>
    public void BridgeBoundsInferOp(IBridgeBoundsInferContext ctx, Op target, IRArray<TIR.Range> output_segment)
    {
        // TODO: Add printers cache.
        var inferencerType = typeof(IBoundInferencer<>).MakeGenericType(target.GetType());
        var inferencer = (IBoundInferencer)_serviceProvider.GetRequiredService(inferencerType);
        inferencer.Visit(ctx, target, output_segment);
    }

    public void BridgeTileStepOp(IBridgeBoundsInferContext ctx, Op target)
    {
        var inferencerType = typeof(IBoundInferencer<>).MakeGenericType(target.GetType());
        var inferencer = (IBoundInferencer)_serviceProvider.GetRequiredService(inferencerType);
        inferencer.VisitTileStep(ctx, target);
    }
}

