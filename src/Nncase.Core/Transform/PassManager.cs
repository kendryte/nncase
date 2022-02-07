// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Transform
{
    /// <summary>
    /// Pass manager.
    /// </summary>
    public class PassManager : IEnumerable<FunctionPass>
    {
        private readonly IRModule _module;
        private readonly RunPassOptions _options;
        private readonly List<FunctionPass> _passes = new List<FunctionPass>();

        /// <summary>
        /// Initializes a new instance of the <see cref="PassManager"/> class.
        /// </summary>
        /// <param name="module">Module.</param>
        /// <param name="options">Options.</param>
        public PassManager(IRModule module, RunPassOptions options)
        {
            _module = module;
            _options = options;
        }

        /// <summary>
        /// Add function pass.
        /// </summary>
        /// <param name="pass">Pass.</param>
        public void Add(FunctionPass pass)
        {
            _passes.Add(pass);
        }
        
        /// <inheritdoc/>
        public IEnumerator<FunctionPass> GetEnumerator()
        {
            return ((IEnumerable<FunctionPass>)_passes).GetEnumerator();
        }

        /// <summary>
        /// Run passes and update the module funciton.
        /// </summary>
        public void Run()
        {
            foreach (var i in Enumerable.Range(0, _module.Functions.Count))
            {
                foreach (var pass in _passes)
                {
                    _module.Update(i, pass.Run(_module.Functions[i], _options));
                }
            }
        }

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)_passes).GetEnumerator();
        }
    }
}
