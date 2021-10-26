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
    public sealed record Tuple(IRArray<Expr> Fields) : Expr, IEnumerable<Expr>, IReadOnlyList<Expr>
    {
        /// <summary>
        /// Void type.
        /// </summary>
        public static readonly Tuple Void = new(ImmutableArray<Expr>.Empty);

        public Tuple(params Expr[] Fields) : this(ImmutableArray.Create<Expr>(Fields)) { }

        public Expr this[int index] => Fields[index];

        public int Count => Fields.Count;

        public IEnumerator<Expr> GetEnumerator()
        {
            return Fields.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)Fields).GetEnumerator();
        }
    }
}
