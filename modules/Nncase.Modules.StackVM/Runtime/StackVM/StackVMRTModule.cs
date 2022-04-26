// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.StackVM;

/// <summary>
/// StackVM runtime module.
/// </summary>
public class StackVMRTModule : RTModule
{
    /// <summary>
    /// StackVM module kind.
    /// </summary>
    public static readonly string Kind = "stackvm";

    /// <summary>
    /// Initializes a new instance of the <see cref="StackVMRTModule"/> class.
    /// </summary>
    /// <param name="functions">Functions.</param>
    public StackVMRTModule(IReadOnlyList<IRTFunction> functions)
        : base(functions)
    {
    }
}
