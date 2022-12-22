// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// The None Expression is a placeholder for optional paramseter.
/// </summary>
public record None : Expr
{
    /// <summary>
    /// The default None expression instance.
    /// </summary>
    public static None Default = new();

    private None()
    {
    }
}
