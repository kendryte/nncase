// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Transform;

/// <summary>
/// EGraph interface.
/// </summary>
public interface IEGraph
{
    /// <summary>
    /// Gets nodes.
    /// </summary>
    IEnumerable<ENode> Nodes { get; }
}
