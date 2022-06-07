// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Nncase.CodeGen;
using Nncase.Transform;

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

    void RegisterTargetDependentPass(PassManager passManager, CompileOptions options);

    void RegisterQuantizePass(PassManager passManager);

    void RegisterTargetDependentAfterQuantPass(PassManager passManager);

    /// <summary>
    /// Create module builder.
    /// </summary>
    /// <param name="moduleKind">Module kind.</param>
    /// <returns>Module builder.</returns>
    IModuleBuilder CreateModuleBuilder(string moduleKind);
}
