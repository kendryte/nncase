// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Nncase.CodeGen;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.Passes;

namespace Nncase.Targets;

public interface INTTModuleCompiler : IModuleCompiler
{
    int Lane { get; }

    int Nr { get; }
}
