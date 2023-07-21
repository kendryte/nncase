using Nncase;
using Nncase.IR;

namespace Nncase.TIR;


/// <summary>
/// the memory type.
/// </summary>
public enum MemoryLocation : byte
{
    /// <summary>
    /// input.
    /// </summary>
    Input = 0,

    /// <summary>
    /// output.
    /// </summary>
    Output = 1,

    /// <summary>
    /// constant data.
    /// </summary>
    Rdata = 2,

    /// <summary>
    /// compute temp data.
    /// </summary>
    Data = 3,

    /// <summary>
    /// shared data.
    /// </summary>
    SharedData = 4,

    /// <summary>
    /// l2 data.
    /// </summary>
    L2Data = 5,

    /// <summary>
    /// L1 data.
    /// </summary>
    L1Data = 6,

    /// <summary>
    /// base addr.
    /// </summary>
    PrivateBase = 64,
}

public sealed class MemSpan : Expr
{
    public MemSpan(Expr size, MemoryLocation location) : base(new[] { None.Default, size })
    {
        Location = location;
    }

    public MemSpan(Expr start, Expr size, MemoryLocation location) : base(new[] { start, size })
    {
        Location = location;
    }

    /// <summary>
    /// Gets the start.
    /// </summary>
    public Expr Start => Operands[0];

    /// <summary>
    /// Gets the size of bytes.
    /// </summary>
    public Expr Size => Operands[1];

    /// <summary>
    /// Gets the memory location.
    /// </summary>
    public MemoryLocation Location { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitMemSpan(this, context);


    public MemSpan With(Expr? start = null, Expr? size = null, MemoryLocation? location = null) => new(start ?? Start, size ?? Size, location ?? Location);
}