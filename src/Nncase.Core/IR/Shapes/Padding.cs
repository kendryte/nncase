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
    }

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
}

public sealed class Paddings : BaseExpr, IEquatable<Paddings?>, IReadOnlyList<Padding>
{
    public static readonly Paddings Empty = new Paddings();

    public Paddings(params Padding[] paddings)
        : base(paddings)
    {
    }

    public ReadOnlySpan<Padding> Values => SpanUtility.UnsafeCast<BaseExpr, Padding>(Operands);

    public int Rank => Values.Length;

    public int Count => Values.Length;

    public override Padding this[Dimension index] => index switch
    {
        DimConst dc => Values[(int)dc.Value],
        _ => throw new ArgumentException("Index must be a constant dimension."),
    };

    public Padding this[int index] => Values[index];

    public static implicit operator Paddings(Padding[] paddings) => new Paddings(paddings);

    public static implicit operator Paddings(Tensor<long> tensor)
    {
        if (tensor.Shape.Rank != 2)
        {
            throw new ArgumentException("Tensor must have 2 dimensions.");
        }

        return new Paddings(
            Enumerable.Range(0, (int)tensor.Dimensions[0])
                .Select(i => new Padding(tensor[i, 0], tensor[i, 1]))
                .ToArray());
    }

    public static implicit operator Paddings(Tensor<int> tensor) => tensor.Cast<long>(CastMode.KDefault);

    public static implicit operator Paddings(int[,] array) => Tensor.From(array);

    public static implicit operator Paddings(long[,] array) => Tensor.From(array);

    public static Paddings Zeros(int rank) => Enumerable.Repeat(Padding.Zero, rank).ToArray();

    public IEnumerator<Padding> GetEnumerator()
    {
        for (int i = 0; i < Count; i++)
        {
            yield return Values[i];
        }
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitPaddings(this, context);

    public Paddings With(Padding[]? values = null) => new Paddings(values ?? Values.ToArray());

    public override string ToString()
    {
        return $"Paddings({StringUtility.Join(", ", Values)})";
    }

    public override bool Equals(object? obj)
    {
        return obj is Paddings paddings && Equals(paddings);
    }

    public bool Equals(Paddings? other)
    {
        if (other is null)
        {
            return false;
        }

        if (Values.Length != other.Values.Length)
        {
            return false;
        }

        for (int i = 0; i < Values.Length; i++)
        {
            if (!Values[i].Equals(other.Values[i]))
            {
                return false;
            }
        }

        return true;
    }

    protected override int GetHashCodeCore()
    {
        var hashCode = default(HashCode);
        foreach (var padding in Values)
        {
            hashCode.Add(padding);
        }

        return hashCode.ToHashCode();
    }
}
