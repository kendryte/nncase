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
    public sealed class IRModule
    {
        private List<Callable> _callables;

        /// <summary>
        /// the index of the entry function.
        /// </summary>
        int _entryIndex;

        /// <summary>
        /// Initializes a new instance of the <see cref="IRModule"/> class.
        /// </summary>
        /// <param name="main"> main func.</param>
        public IRModule(Callable main)
        {
            _callables = new();
            _callables.Add(main);
            _entryIndex = 0;
        }

        /// <summary>
        /// the default IrModule ctor.
        /// </summary>
        public IRModule()
        {
            _callables = new();
            _entryIndex = -1;
        }

        /// <summary>
        /// Gets functions.
        /// </summary>
        public IReadOnlyList<Callable> Callables => _callables;

        /// <summary>
        /// Gets or sets entry function.
        /// </summary>
        public Callable? Entry
        {
            get => _entryIndex == -1 ? null : Callables[_entryIndex];
            set
            {
                if (value is null) _entryIndex = -1;
                else
                {
                    _entryIndex = _callables.IndexOf(value);
                    if (_entryIndex == -1)
                    {
                        _callables.Add(value);
                        _entryIndex = _callables.Count - 1;
                    }
                }
            }
        }

        /// <summary>
        /// Add function.
        /// </summary>
        /// <param name="function">Callable to add.</param>
        public void Add(Callable function)
        {
            _callables.Add(function);
        }

        /// <summary>
        /// update the entry function defination.
        /// </summary>
        /// <param name="i">function index.</param>
        /// <param name="function">the entry function defination.</param>
        public void Update(int i, Callable function)
        {
            _callables.RemoveAt(i);
            _callables.Add(function);
        }

        /// <summary>
        /// schedule result.
        /// </summary>
        public Schedule.SchedModuleResult? SchedResult;
    }
}
