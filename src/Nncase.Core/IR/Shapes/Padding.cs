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

public sealed partial class Padding : Expr, IEquatable<Padding?>
{
    public static readonly Padding Zero = new Padding(Dimension.Zero, Dimension.Zero);

    public Padding(Dimension before, Dimension after)
        : base([before, after])
    {
    }

    public Dimension Before => (Dimension)Operands[0];

    public Dimension After => (Dimension)Operands[1];

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

public sealed class Paddings : Expr, IEquatable<Paddings?>, IReadOnlyList<Padding>
{
    public static readonly Paddings Empty = new Paddings();

    public Paddings(params Padding[] paddings)
        : base(paddings)
    {
    }

    public ReadOnlySpan<Padding> Values => SpanUtility.UnsafeCast<Expr, Padding>(Operands);

    public int Count => Operands.Length;

    public new Padding this[int index] => Values[index];

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
