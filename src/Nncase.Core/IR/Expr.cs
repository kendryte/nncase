// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Nncase.IR
{
    /// <summary>
    /// Expression.
    /// </summary>
    public abstract partial record Expr
    {

        protected int? _hashcode;

        /// <summary>
        /// Gets or sets checked type.
        /// </summary>
        public IRType? CheckedType { get; set; }

        public virtual Shape CheckedShape => CheckedType switch
        {
            TensorType type => type.Shape,
            _ => throw new InvalidOperationException("Only The Expr Have CheckedType Can Get It's Shape")
        };

        public DataType CheckedDataType => CheckedType switch
        {
            // todo:more info
            TensorType type => type.DType,
            _ => throw new InvalidOperationException("Expr don't have a valid tensor type")
        };
        
        public virtual int Rank => CheckedShape.Rank;

        public virtual bool Equals(Expr? other)
        {
            return !(other is null) && EqualityContract == other.EqualityContract;
        }

        public override int GetHashCode()
        {
            return _hashcode ??= EqualityComparer<Type>.Default.GetHashCode(EqualityContract);
        }

        public override string ToString()
        {
            var builder = new StringBuilder();
            var writer = new StringWriter(builder);
            IRPrinter.DumpExprAsIL(writer, this);
            return builder.ToString();
        }
    }
}
