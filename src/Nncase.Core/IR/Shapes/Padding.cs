// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Passes;
using Nncase.Passes.Mutators;
using Nncase.Utilities;

namespace Nncase.IR.Shapes;

public sealed partial class Padding : BaseExpr, IEquatable<Padding?>
{
    public static readonly Padding Zero = new Padding(Dimension.Zero, Dimension.Zero);

    public Padding(Dimension before, Dimension after)
        : base([before, after])
    {
        RefreshKind();
    }

    /// <summary>
    /// Gets kind.
    /// </summary>
    public ShapeKind Kind { get; private set; }

    /// <summary>
    /// Gets a value indicating whether fixed.
    /// </summary>
    public bool IsFixed => Kind == ShapeKind.Fixed;

    /// <summary>
    /// Gets a value indicating whether has unknown dimension.
    /// </summary>
    public bool HasUnknownDimension => Kind == ShapeKind.HasUnknownDimension;

    public Dimension Before => (Dimension)Operands[0];

    public Dimension After => (Dimension)Operands[1];

    public override BaseExpr this[Dimension index] => throw new NotSupportedException();

    public static implicit operator Padding(Tensor<long> tensor)
    {
        if (tensor.Shape.Rank != 1 && tensor.Shape[0] != 2)
        {
            throw new ArgumentException("Tensor must have 1 dimension with size 2.");
        }

        return new Padding(tensor[0], tensor[1]);
    }

    public static implicit operator Padding(Dimension[] array)
    {
        if (array.Length != 2)
        {
            throw new ArgumentException("Array must have length 2.");
        }

        return new Padding(array[0], array[1]);
    }

    public static implicit operator Padding(Tensor<int> tensor) => tensor.Cast<long>(CastMode.KDefault);

    public static implicit operator Padding(int[] array) => Tensor.From(array);

    public static implicit operator Padding(long[] array) => Tensor.From(array);

    public static Padding operator +(Padding lhs, Padding rhs) => new Padding(lhs.Before + rhs.Before, lhs.After + rhs.After);

    public Dimension Sum() => Before + After;

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitPadding(this, context);

    public Padding With(Dimension? before = null, Dimension? after = null) => new Padding(before ?? Before, after ?? After);

    public void Deconstruct(out Dimension before, out Dimension after)
    {
        before = Before;
        after = After;
    }

    public override string ToString()
    {
        return $"Padding({Before}, {After})";
    }

    public long[] ToValueArray() => [Before.FixedValue, After.FixedValue];

    public override bool Equals(object? obj)
    {
        return obj is Padding padding && Equals(padding);
    }

    public bool Equals(Padding? other)
    {
        if (other is null)
        {
            return false;
        }

        return Before == other.Before && After == other.After;
    }

    protected override int GetHashCodeCore()
    {
        return HashCode.Combine(Before, After);
    }

    protected override void OnOperandsReplaced()
    {
        base.OnOperandsReplaced();
        RefreshKind();
    }

    private void RefreshKind()
    {
        Kind = Before.IsFixed && After.IsFixed ? ShapeKind.Fixed : ShapeKind.HasUnknownDimension;
    }
}
