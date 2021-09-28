// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Module.
    /// </summary>
    public sealed class Module
    {
        private List<Function> _functions;

        /// <summary>
        /// Initializes a new instance of the <see cref="Module"/> class.
        /// </summary>
        public Module()
        {
            _functions = new List<Function>();
        }

        /// <summary>
        /// Gets functions.
        /// </summary>
        public IReadOnlyList<Function> Functions => _functions;

        /// <summary>
        /// Gets or sets entry function.
        /// </summary>
        public Function? Entry { get; set; }

        /// <summary>
        /// Add function.
        /// </summary>
        /// <param name="function">Function to add.</param>
        public void Add(Function function)
        {
            _functions.Add(function);
        }
    }
}
