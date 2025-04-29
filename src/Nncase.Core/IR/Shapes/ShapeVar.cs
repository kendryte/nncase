// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.IR;

public sealed class ShapeVar : Shape, IVar, IEquatable<ShapeVar?>
{
    private static int _globalVarIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="ShapeVar"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    /// <param name="rank">Rank.</param>
    public ShapeVar(string name, int rank)
        : base(Array.Empty<Expr>())
    {
        GlobalVarIndex = GetNextId();
        Name = name;
        Rank = rank;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ShapeVar"/> class.
    /// </summary>
    /// <param name="rank">Rank.</param>
    public ShapeVar(int rank)
        : base(Array.Empty<Expr>())
    {
        GlobalVarIndex = GetNextId();
        Name = $"shapeVar_{GlobalVarIndex}";
        Rank = rank;
    }

    /// <summary>
    /// Gets the global var index.
    /// </summary>
    public int GlobalVarIndex { get; }

    /// <summary>
    /// Gets name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the rank.
    /// </summary>
    public override int Rank { get; }

    public override ShapeKind Kind => ShapeKind.HasUnknownDimension;

    public override Dimension this[Dimension index] => new DimAt(this, Dimension.Positive(index, Rank));

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitShapeVar(this, context);

    public ShapeVar With(string? name = null, int? rank = null) => new ShapeVar(name ?? Name, rank ?? Rank)
    {
        Metadata =
        {
            Range = Metadata.Range,
        },
    };

    IVar IVar.With(string? name) => With(name);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as ShapeVar);

    /// <inheritdoc/>
    public bool Equals(ShapeVar? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && GlobalVarIndex == other.GlobalVarIndex;
    }

    bool IEquatable<IVar?>.Equals(IVar? other) => Equals(other as ShapeVar);

    public override string ToString() => $"{Name}";

    public override bool IsAssignableFrom(Shape shape) => shape is RankedShape rankedShape && rankedShape.Rank == Rank;

    public override Expr ToValueArrayExpr() => IR.F.Shapes.AsTensor(this);

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(GlobalVarIndex);

    private static int GetNextId()
    {
        return Interlocked.Increment(ref _globalVarIndex);
    }
}
