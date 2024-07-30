// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// the memory type.
/// </summary>
[Flags]
public enum MemoryLocation
{
    /// <summary>
    /// input.
    /// </summary>
    Input = 1 << 1,

    /// <summary>
    /// output.
    /// </summary>
    Output = 1 << 2,

    /// <summary>
    /// constant data.
    /// </summary>
    Rdata = 1 << 3,

    /// <summary>
    /// compute temp data.
    /// </summary>
    Data = 1 << 4,

    /// <summary>
    /// shared data.
    /// </summary>
    SharedData = 1 << 5,

    /// <summary>
    /// l2 data.
    /// </summary>
    L2Data = 1 << 6,

    /// <summary>
    /// L1 data.
    /// </summary>
    L1Data = 1 << 7,

    /// <summary>
    /// base addr.
    /// </summary>
    PrivateBase = 1 << 8,
}

public sealed class MemSpan : Expr
{
    public MemSpan(Expr size, MemoryLocation location, int hierarchy = 0)
        : base(new[] { None.Default, size })
    {
        Location = location;
        Hierarchy = hierarchy;
    }

    public MemSpan(Expr start, Expr size, MemoryLocation location, int hierarchy = 0)
        : base(new[] { start, size })
    {
        Location = location;
        Hierarchy = hierarchy;
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

    /// <summary>
    /// Gets the memory hierarchy.
    /// </summary>
    public int Hierarchy { get; }

    public MemSpan SubSpan(Expr offset, Expr size) => new MemSpan((Start is None ? IR.F.Buffer.DDrOf(this) : Start) + offset, size, Location);

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitMemSpan(this, context);

    public MemSpan With(Expr? start = null, Expr? size = null, MemoryLocation? location = null, int? hierarchy = null) => new(start ?? Start, size ?? Size, location ?? Location, hierarchy ?? Hierarchy);

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        if (ReferenceEquals(this, obj))
        {
            return true;
        }

        return obj is MemSpan other && GetHashCode() == other.GetHashCode() && Location == other.Location && Operands.SequenceEqual(other.Operands);
    }

    protected override int GetHashCodeCore() => HashCode.Combine(Location, base.GetHashCodeCore());
}
