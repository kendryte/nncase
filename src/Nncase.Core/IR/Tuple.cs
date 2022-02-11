// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Tuple expression.
    /// </summary>
    public sealed record Tuple(IRArray<Expr> Fields) : Expr, ITuple
    {
        /// <summary>
        /// Void type.
        /// </summary>
        public static readonly TupleConst Void = new(ImmutableArray<Const>.Empty);

        public Tuple(params Expr[] Fields) : this(ImmutableArray.Create<Expr>(Fields)) { }

        public Tuple(IEnumerable<Expr> Fields) : this(Fields.ToArray()) { }

        /// <inheritdoc/>
        public int Count => Fields.Count;

        IReadOnlyList<Expr> ITuple.Fields => Fields;

        /// <inheritdoc/>
        public Expr this[int index] => Fields[index];

        /// <summary>
        /// cast the value tuple to ir array.
        /// </summary>
        public static implicit operator Tuple(ValueTuple<Expr> tuple) =>
            new Tuple(ImmutableArray.Create(new Expr[] { tuple.Item1 }));

        /// <summary>
        /// cast the value tuple to ir array.
        /// </summary>
        public static implicit operator Tuple(ValueTuple<Expr, Expr> tuple) =>
            new Tuple(ImmutableArray.Create(new Expr[] { tuple.Item1, tuple.Item2 }));

        /// <summary>
        /// cast the value tuple to ir array.
        /// </summary>
        public static implicit operator Tuple(ValueTuple<Expr, Expr, Expr> tuple) =>
            new Tuple(ImmutableArray.Create(new Expr[] { tuple.Item1, tuple.Item2, tuple.Item3 }));

        /// <summary>
        /// cast the value tuple to ir array.
        /// </summary>
        public static implicit operator Tuple(ValueTuple<Expr, Expr, Expr, Expr> tuple) =>
            new Tuple(ImmutableArray.Create(new Expr[] { tuple.Item1, tuple.Item2, tuple.Item3, tuple.Item4 }));

        /// <inheritdoc/>
        public IEnumerator<Expr> GetEnumerator()
        {
            return Fields.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
