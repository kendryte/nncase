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
    public class PassManager : IEnumerable<BasePass>
    {
        private readonly IRModule _module;
        private readonly RunPassOptions _options;
        private readonly List<BasePass> _passes = new List<BasePass>();

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
        public void Add(BasePass pass)
        {
            _passes.Add(pass);
        }

        /// <inheritdoc/>
        public IEnumerator<BasePass> GetEnumerator()
        {
            return ((IEnumerable<BasePass>)_passes).GetEnumerator();
        }

        /// <summary>
        /// Run passes and update the module.
        /// </summary>
        public async Task RunAsync()
        {
            var passes = _passes;
            while (passes.Count() > 0)
            {
                var type = _passes.First().GetType();
                var candiate = passes.TakeWhile(item => item.GetType().TypeHandle.Equals(type.TypeHandle));
                passes.Skip(candiate.Count());
                if (type == typeof(FunctionPass))
                    await runFunctionAsync(candiate);
                else if (type == typeof(ModulePass))
                    await runModuleAsync(candiate);
            }
        }

        private async Task runFunctionAsync(IEnumerable<BasePass> passes)
        {
            int i = 0;
            while (i < _module.Functions.Count)
            {
                foreach (var pass in passes)
                {
                    var post = await ((FunctionPass)pass).RunAsync(_module.Functions[i], _options);
                    if (post is PrimFunctionWrapper wrapper)
                    {
                        _module.Add(wrapper.Target);
                    }
                    _module.Update(i, post);
                }
                i++;
            }
        }

        private async Task runModuleAsync(IEnumerable<BasePass> passes)
        {
            foreach (var pass in passes)
            {
                await ((ModulePass)pass).RunAsync(_module, _options);
            }
        }

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)_passes).GetEnumerator();
        }
    }
}
