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
        /// <summary>
        /// Gets or sets checked type.
        /// </summary>
        public IRType? CheckedType { get; set; }

        public override string ToString()
        {
            var builder = new StringBuilder();
            var writer = new StringWriter(builder);
            IRPrinter.DumpExprAsIL(writer, this);
            return builder.ToString();
        }

    }
}
