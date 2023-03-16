// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Tuple interface.
/// </summary>
public interface ITuple
{
    /// <summary>
    /// Gets fields count.
    /// </summary>
    int Count { get; }
}
