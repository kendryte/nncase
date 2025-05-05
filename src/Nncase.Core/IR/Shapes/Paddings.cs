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

public sealed class Paddings : BaseExpr, IEquatable<Paddings?>, IReadOnlyList<Padding>
{
    public static readonly Paddings Empty = new Paddings();

    public Paddings(params Padding[] paddings)
        : base(paddings)
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

    public long[,] ToValueArray()
    {
        if (IsFixed)
        {
            var result = new long[Rank, 2];
            for (int i = 0; i < Rank; i++)
            {
                result[i, 0] = Values[i].Before.FixedValue;
                result[i, 1] = Values[i].After.FixedValue;
            }

            return result;
        }
        else
        {
            throw new InvalidOperationException("Cannot convert to value array when paddings are not fixed.");
        }
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

    protected override void OnOperandsReplaced()
    {
        base.OnOperandsReplaced();
        RefreshKind();
    }

    private void RefreshKind()
    {
        Kind = Values.AsValueEnumerable().All(x => x.IsFixed) ? ShapeKind.Fixed : ShapeKind.HasUnknownDimension;
    }
}
