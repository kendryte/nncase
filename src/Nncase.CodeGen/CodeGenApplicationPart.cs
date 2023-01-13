﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Nncase.CodeGen;
using Nncase.Converters;
using Nncase.Diagnostics;
using Nncase.Hosting;

namespace Nncase;

/// <summary>
/// CodeGen application part extensions.
/// </summary>
public static class CodeGenApplicationPart
{
    /// <summary>
    /// Add diagnostics assembly.
    /// </summary>
    /// <param name="registrator">Service registrator.</param>
    /// <returns>Configured service registrator.</returns>
    public static IRegistrator AddCodeGen(this IRegistrator registrator)
    {
        return registrator.RegisterModule<CodeGenModule>();
    }
}
