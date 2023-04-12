// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc;

namespace Nncase.Compiler.Hosting;

/// <summary>
/// Compiler builder.
/// </summary>
public interface ICompilerBuilder
{
    /// <summary>
    /// Configure modules.
    /// </summary>
    /// <param name="configureModules">Configure modules action.</param>
    /// <returns>Compiler builder.</returns>
    ICompilerBuilder ConfigureModules(Action<IRegistrator> configureModules);
}

internal sealed class CompilerBuilder : ICompilerBuilder
{
    private readonly IContainer _container;

    public CompilerBuilder(IContainer container)
    {
        _container = container;
    }

    public ICompilerBuilder ConfigureModules(Action<IRegistrator> configureModules)
    {
        configureModules(_container);
        return this;
    }
}
