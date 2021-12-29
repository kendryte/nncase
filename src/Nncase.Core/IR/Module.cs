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
        /// <param name="main"> main func</param>
        public Module(Function main)
        {
            _functions = new();
            _functions.Add(main);
        }

        public Module()
        {
            _functions = new();
        }

        /// <summary>
        /// Gets functions.
        /// </summary>
        public IReadOnlyList<Function> Functions => _functions;

        /// <summary>
        /// Gets or sets entry function.
        /// </summary>
        public Function? Entry
        {
            get => _functions.Count > 0 ? _functions.First() : null;
            set
            {
                if (value is not null)
                {
                    if (_functions.Count > 0)
                        _functions[0] = value;
                    else
                        _functions.Add(value);
                }
            }
        }

        /// <summary>
        /// Add function.
        /// </summary>
        /// <param name="function">Function to add.</param>
        public void Add(Function function)
        {
            _functions.Add(function);
        }

        /// <summary>
        /// update the entry function defination
        /// </summary>
        /// <param name="i">function index.</param>
        /// <param name="function">the entry function defination.</param>
        public void Update(int i, Function function)
        {
            _functions.RemoveAt(i);
            _functions.Add(function);
        }
    }
}
