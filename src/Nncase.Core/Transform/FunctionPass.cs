// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Transform
{
    /// <summary>
    /// Pass in Callable scope.
    /// </summary>
    public abstract class FunctionPass
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FunctionPass"/> class.
        /// </summary>
        /// <param name="name">Name.</param>
        public FunctionPass(string name)
        {
            Name = name;
        }

        /// <summary>
        /// Gets name.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Run current pass for specific function.
        /// </summary>
        /// <param name="callable">Target function.</param>
        /// <param name="options">Options.</param>
        public Callable Run(Callable callable, RunPassOptions options)
            => RunCore(callable, options);

        /// <summary>
        /// Run pass implementation for derived class.
        /// </summary>
        /// <param name="callable">Target function.</param>
        /// <param name="options">Options.</param>
        protected abstract Callable RunCore(Callable callable, RunPassOptions options);

        /// <summary>
        /// the callback function you can custom process func with run pass options.
        /// </summary>
        /// <param name="callable"> func without run pass.</param>
        /// <param name="options"></param>
        protected virtual void OnPassStart(Callable callable, RunPassOptions options)
        {
        }

        /// <summary>
        /// the callback function you can custom process func with run pass options.
        /// </summary>
        /// <param name="callable"> func with rewrited. </param>
        /// <param name="options"></param>
        protected virtual void OnPassEnd(Callable callable, RunPassOptions options)
        { }
    }
}
