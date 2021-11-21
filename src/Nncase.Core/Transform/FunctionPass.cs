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
    /// Pass in function scope.
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
        /// Run pass.
        /// </summary>
        /// <param name="function">Target function.</param>
        /// <param name="options">Options.</param>
        public Function Run(Function function, RunPassOptions options)
            => RunCore(function, options);


        /// <summary>
        /// Run pass implementation for derived class.
        /// </summary>
        /// <param name="function">Target function.</param>
        /// <param name="options">Options.</param>
        protected abstract Function RunCore(Function function, RunPassOptions options);
    }
}
