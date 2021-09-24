// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Function expression.
    /// </summary>
    public sealed class Function : Expression
    {
        private static int _globalFuncIndex = 0;

        /// <summary>
        /// Initializes a new instance of the <see cref="Function"/> class.
        /// </summary>
        /// <param name="name">Name.</param>
        /// <param name="parameters">Parameters.</param>
        /// <param name="body">Body.</param>
        public Function(string name, IReadOnlyList<Expression> parameters, Expression body)
        {
            Name = name;
            Parameters = parameters;
            Body = body;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Function"/> class.
        /// </summary>
        /// <param name="parameters">Parameters.</param>
        /// <param name="body">Body.</param>
        public Function(IReadOnlyList<Expression> parameters, Expression body)
            : this($"func_{_globalFuncIndex++}", parameters, body)
        {
        }

        /// <summary>
        /// Gets or sets name of the function.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets parameters of the function.
        /// </summary>
        public IReadOnlyList<Expression> Parameters { get; }

        /// <summary>
        /// Gets or sets the body of the function.
        /// </summary>
        public Expression Body { get; set; }
    }
}
