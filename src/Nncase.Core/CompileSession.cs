// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Microsoft.Extensions.DependencyInjection;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Transform;

namespace Nncase;

/// <summary>
/// Compile session.
/// </summary>
public sealed class CompileSession : IDisposable
{
    private readonly IResolverContext _serviceProvider;

    private bool _disposedValue;
    private ICompiler? _compiler;
    private IDumpperFactory? _dumpperFactory;
    private IModelBuilder? _modelBuilder;

    /// <summary>
    /// Initializes a new instance of the <see cref="CompileSession"/> class.
    /// </summary>
    /// <param name="serviceProvider">Service provider.</param>
    /// <param name="target">Target.</param>
    /// <param name="compileOptions">Compile options.</param>
    internal CompileSession(IResolverContext serviceProvider, ITarget target, CompileOptions compileOptions)
    {
        _serviceProvider = serviceProvider;
        Target = target;
        CompileOptions = compileOptions;
    }

    /// <summary>
    /// Gets service provider.
    /// </summary>
    public IServiceProvider ServiceProvider => _serviceProvider;

    /// <summary>
    /// Gets target.
    /// </summary>
    public ITarget Target { get; }

    /// <summary>
    /// Gets compile options.
    /// </summary>
    public CompileOptions CompileOptions { get; }

    /// <summary>
    /// Gets compiler.
    /// </summary>
    public ICompiler Compiler => _compiler ??= ServiceProvider.GetRequiredService<ICompiler>();

    /// <summary>
    /// Gets model builder.
    /// </summary>
    public IModelBuilder ModelBuilder => _modelBuilder ??= ServiceProvider.GetRequiredService<IModelBuilder>();

    /// <summary>
    /// Gets dumpper factory.
    /// </summary>
    public IDumpperFactory DumpperFactory => _dumpperFactory ??= ServiceProvider.GetRequiredService<IDumpperFactory>();

    /// <summary>
    /// Create new compile session.
    /// </summary>
    /// <param name="target">Compile target.</param>
    /// <param name="compileOptions">Compile options.</param>
    /// <returns>Created compile session.</returns>
    public static CompileSession Create(ITarget target, CompileOptions compileOptions)
    {
        var rootContainer = (IContainer)CompilerServices.ServiceProvider;
        var childContainer = rootContainer.CreateChild(RegistrySharing.CloneButKeepCache, new object());
        childContainer.RegisterInstance(target);
        childContainer.RegisterInstance(compileOptions);

        var session = new CompileSession(childContainer, target, compileOptions);
        childContainer.RegisterInstance(session);
        return session;
    }

    /// <summary>
    /// Create new pass manager.
    /// </summary>
    /// <param name="name">Name.</param>
    /// <returns>Created pass manager.</returns>
    public PassManager CreatePassManager(string name)
    {
        return ActivatorUtilities.CreateInstance<PassManager>(ServiceProvider, name);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
    }

    private void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            if (disposing)
            {
                _serviceProvider.Dispose();
            }

            _disposedValue = true;
        }
    }
}
