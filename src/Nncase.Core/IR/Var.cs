// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Variable expression.
/// </summary>
public sealed class Var : Expr, IEquatable<Var?>
{
    private static int _globalVarIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="Var"/> class.
    /// ctor.
    /// </summary>
    public Var(string name, IRType typeAnnotation)
        : base(Array.Empty<Expr>())
    {
        TypeAnnotation = typeAnnotation;
        CheckedType = TypeAnnotation;
        GlobalVarIndex = GetNextId();
        Name = name;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Var"/> class.
    /// </summary>
    /// <param name="typeAnnotation">Type annotation.</param>
    public Var(IRType typeAnnotation)
        : base(Array.Empty<Expr>())
    {
        TypeAnnotation = typeAnnotation;
        CheckedType = TypeAnnotation;
        GlobalVarIndex = GetNextId();
        Name = $"var_{GlobalVarIndex}";
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Var"/> class.
    /// <see cref="Var"/>.
    /// </summary>
    public Var(string name)
        : this(name, AnyType.Default)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Var"/> class.
    /// </summary>
    public Var()
        : this(AnyType.Default)
    {
    }

    /// <summary>
    /// Gets the global var index.
    /// </summary>
    public int GlobalVarIndex { get; }

    /// <summary>
    /// Gets name.
    /// </summary>
    public string Name { get; init; }

    /// <summary>
    /// Gets typeAnnotation.
    /// </summary>
    public IRType TypeAnnotation { get; init; }

    /// <summary>
    /// get any var.
    /// </summary>
    public static implicit operator Var(string name) => new Var(name, AnyType.Default);

    public static bool operator ==(Var? left, Var? right) => EqualityComparer<Var>.Default.Equals(left, right);

    public static bool operator !=(Var? left, Var? right) => !(left == right);

    /// <summary>
    /// get scalar var.
    /// </summary>
    public static Var Scalar(string name, DataType dtype) => new Var(name, new TensorType(dtype, Shape.Scalar));

    /// <summary>
    /// get handle var.
    /// </summary>
    /// <returns> var. </returns>
    public static Var Handle(string name, DataType dtype, string scope = "") => new Var(name, TensorType.Scalar(new PointerType(dtype, Shape.Scalar)));

    /// <summary>
    /// get the size var. it can be used in tensor shape. like n>=0, m>=0.
    /// </summary>
    public static Var SizeVar(string name) => Scalar(name, DataTypes.Int32);

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitVar(this, context);

    public Var With(string? name = null, IRType? typeAnnotation = null) => new Var(name ?? Name, typeAnnotation ?? TypeAnnotation);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as Var);

    /// <inheritdoc/>
    public bool Equals(Var? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && GlobalVarIndex == other.GlobalVarIndex;
    }

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(GlobalVarIndex);

    private static int GetNextId()
    {
        return Interlocked.Increment(ref _globalVarIndex);
    }
}
