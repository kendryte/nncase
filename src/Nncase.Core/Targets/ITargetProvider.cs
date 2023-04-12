// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Targets;

/// <summary>
/// Target provider.
/// </summary>
public interface ITargetProvider
{
    /// <summary>
    /// Get target.
    /// </summary>
    /// <param name="name">Target name.</param>
    /// <returns>Target.</returns>
    ITarget GetTarget(string name);
}
