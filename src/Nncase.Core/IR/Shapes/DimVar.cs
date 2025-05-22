// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR;

public sealed class DimVar : OpaqueDim, IVar, IEquatable<DimVar?>
{
    private static int _globalVarIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="DimVar"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    public DimVar(string name)
        : base(Array.Empty<Expr>())
    {
        GlobalVarIndex = GetNextId();
        Name = name;
        Metadata.Range = ValueRange<double>.Full;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DimVar"/> class.
    /// </summary>
    public DimVar()
        : base(Array.Empty<Expr>())
    {
        GlobalVarIndex = GetNextId();
        Name = $"dimVar_{GlobalVarIndex}";
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets the global var index.
    /// </summary>
    public int GlobalVarIndex { get; }

    /// <summary>
    /// Gets name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Create a dim var.
    /// </summary>
    public static implicit operator DimVar(string name) => new DimVar(name);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimVar(this, context);

    public DimVar With(string? name = null) => new DimVar(name ?? Name)
    {
        Metadata =
        {
            Range = Metadata.Range,
        },
    };

    IVar IVar.With(string? name) => With(name);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimVar);

    /// <inheritdoc/>
    public bool Equals(DimVar? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && GlobalVarIndex == other.GlobalVarIndex;
    }

    bool IEquatable<IVar?>.Equals(IVar? other) => Equals(other as DimVar);

    public override string ToString() => $"{Name}";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(GlobalVarIndex);

    private static int GetNextId()
    {
        return Interlocked.Increment(ref _globalVarIndex);
    }
}
