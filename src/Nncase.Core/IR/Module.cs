﻿// Copyright (c) Canaan Inc. All rights reserved.
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
    public sealed class IRModule
    {
        private List<BaseFunction> _functions;

        /// <summary>
        /// the index of the entry function.
        /// </summary>
        int _entryIndex;

        /// <summary>
        /// Initializes a new instance of the <see cref="IRModule"/> class.
        /// </summary>
        /// <param name="main"> main func.</param>
        public IRModule(BaseFunction main)
        {
            _functions = new();
            _functions.Add(main);
            _entryIndex = 0;
        }

        /// <summary>
        /// the default IrModule ctor.
        /// </summary>
        public IRModule()
        {
            _functions = new();
            _entryIndex = -1;
        }

        /// <summary>
        /// Gets functions.
        /// </summary>
        public IReadOnlyList<BaseFunction> Functions => _functions;

        /// <summary>
        /// Gets or sets entry function.
        /// </summary>
        public BaseFunction? Entry
        {
            get => _entryIndex == -1 ? null : Functions[_entryIndex];
            set
            {
                if (value is null)
                {
                    _entryIndex = -1;
                }
                else
                {
                    _entryIndex = _functions.IndexOf(value);
                    if (_entryIndex == -1)
                    {
                        _functions.Add(value);
                        _entryIndex = _functions.Count - 1;
                    }
                }
            }
        }

        /// <summary>
        /// Add function.
        /// </summary>
        /// <param name="function">Callable to add.</param>
        public void Add(BaseFunction function)
        {
            _functions.Add(function);
        }

        /// <summary>
        /// update the entry function defination.
        /// </summary>
        /// <param name="i">function index.</param>
        /// <param name="function">the entry function defination.</param>
        public void Update(int i, BaseFunction function)
        {
            _functions[i] = function;
        }
    }
}
