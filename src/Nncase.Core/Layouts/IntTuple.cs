// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;

namespace Nncase.Layouts;

public abstract record RecursiveValue
{
    public static RecursiveValue UnderScore => new UnderScoreValue();

    public static implicit operator RecursiveValue(long value) => new IntValue(value);

    public static implicit operator RecursiveValue(int value) => new IntValue(value);
}

public sealed record UnderScoreValue() : RecursiveValue
{
    public override string ToString() => "_";
}

public sealed record IntValue(long Value) : RecursiveValue
{
    public override string ToString() => Value.ToString();
}

[CollectionBuilder(typeof(TupleBuilder), nameof(TupleBuilder.Create))]
public sealed record CollectValue : RecursiveValue, IReadOnlyList<RecursiveValue>
{
    public CollectValue(IEnumerable<RecursiveValue> elements)
    {
        Elements = elements.ToImmutableArray();
    }

    public ImmutableArray<RecursiveValue> Elements { get; }

    public int Count => Elements.Length;

    public RecursiveValue this[int index] => Elements[index];

    public CollectValue this[Range range]
    {
        get
        {
            var (start, end) = (range.Start.GetOffset(Count), range.End.GetOffset(Count));
            return new CollectValue(Enumerable.Range(start, end - start).Select(i => Elements[i]));
        }
    }

    public IEnumerator<RecursiveValue> GetEnumerator() => ((IEnumerable<RecursiveValue>)Elements).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)Elements).GetEnumerator();

    public override string ToString() => Count switch
    {
        > 1 => $"({string.Join(", ", Elements)})",
        1 => $"({Elements[0]},)",
        0 => "(,)",
        _ => throw new InvalidOperationException(),
    };
}

public static class TupleBuilder
{
    public static CollectValue Create(ReadOnlySpan<RecursiveValue> values) => new CollectValue(values.ToArray());
}
