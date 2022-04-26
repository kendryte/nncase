// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Nncase.CodeGen;

namespace Nncase;

/// <summary>
/// Target.
/// </summary>
public interface ITarget
{
    /// <summary>
    /// Gets target kind.
    /// </summary>
    string Kind { get; }

    /// <summary>
    /// Create module builder.
    /// </summary>
    /// <param name="moduleKind">Module kind.</param>
    /// <returns>Module builder.</returns>
    IModuleBuilder CreateModuleBuilder(string moduleKind);
}
