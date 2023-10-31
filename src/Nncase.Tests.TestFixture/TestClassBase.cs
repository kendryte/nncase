// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Diagnostics;
using Nncase.Targets;
using Xunit.Abstractions;
using Xunit.DependencyInjection;

namespace Nncase.Tests;

/// <summary>
/// Test base class.
/// </summary>
public abstract class TestClassBase : IDisposable
{
    public static readonly string DefaultCompileOptionsKey = "default";

    private CompileSession? _compileSession;
    private IServiceProvider? _serviceScope;
    private CompileSessionScope _compileSessionScope;
    private bool _disposedValue;

    public TestClassBase()
    {
        CompileOptions = new();
    }

    /// <summary>
    /// Gets or sets default target name.
    /// </summary>
    public virtual string DefaultTargetName { get; set; } = "cpu";

    /// <summary>
    /// Gets test output root.
    /// </summary>
    internal static string TestOutputRoot { get; } = Path.Join(SolutionDirectory, "tests_output");

    /// <summary>
    /// Gets nncase solution root directory.
    /// </summary>
    protected static string SolutionDirectory => GetSolutionDirectoryCore();

    /// <summary>
    /// Gets <see cref="Nncase.CompileOptions"/> for current test method.
    /// </summary>
    protected CompileOptions CompileOptions { get; }

    /// <summary>
    /// Gets <see cref="Nncase.CompileSession"/> for current test method.
    /// </summary>
    protected CompileSession CompileSession => _compileSession ?? throw new InvalidOperationException($"{nameof(CompileSession)} is not setup yet, call {nameof(SetupTestMethod)} with initSession = true");

    /// <summary>
    /// Gets <see cref="IDumpper"/> for current test method.
    /// </summary>
    protected IDumpper Dumpper => _serviceScope?.GetRequiredService<IDumpperFactory>().Root ?? throw new InvalidOperationException($"Test is not setup yet, call {nameof(SetupTestMethod)}");

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Setup test method.
    /// </summary>
    /// <param name="initSession">Initialize <see cref="CompileSession"/>.</param>
    /// <param name="targetName">Target name, default to <see cref="DefaultTargetName"/>.</param>
    /// <param name="methodName">Method name.</param>
    protected internal void SetupTestMethod(bool initSession = true, string? targetName = null, [CallerMemberName] string? methodName = null)
    {
        CompileOptions.DumpDir = Path.Join(TestOutputRoot, GetType().Name, methodName);
        if (initSession)
        {
            var target = CompilerServices.GetTarget(targetName ?? DefaultTargetName);
            _compileSession = CompileSession.Create(target, CompileOptions);
            _serviceScope = _compileSession;
            _compileSessionScope = new(_compileSession);
        }
        else
        {
            var childContainer = CompilerServices.CreateScope();
            childContainer.RegisterInstance(CompileOptions);
            _serviceScope = childContainer;
        }
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            if (disposing)
            {
                _compileSessionScope.Dispose();
                ((IDisposable?)_serviceScope)?.Dispose();
            }

            _disposedValue = true;
        }
    }

    private static string GetSolutionDirectoryCore([CallerFilePath] string? callerFilePath = null)
    {
        return Path.GetFullPath(Path.Join(callerFilePath!, "../../.."));
    }
}
