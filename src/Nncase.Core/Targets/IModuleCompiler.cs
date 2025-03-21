// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CodeGen;
using Nncase.IR;

namespace Nncase.Targets;

public interface IModuleCompiler
{
    string ModuleKind { get; }

    bool IsSupportedCall(Call call, CompileOptions options);

    /// <summary>
    /// Create module builder.
    /// </summary>
    /// <param name="options">compile options.</param>
    /// <returns>Module builder.</returns>
    IModuleBuilder CreateModuleBuilder(CompileOptions options);
}
