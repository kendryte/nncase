// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

public struct CompileSessionScope : IDisposable
{
    private static readonly AsyncLocal<CompileSession?> _compileSession = new AsyncLocal<CompileSession?>();

    private readonly bool _initialized;
    private readonly CompileSession? _originalCompileSession;

    public CompileSessionScope(CompileSession compileSession)
    {
        _initialized = true;
        _originalCompileSession = _compileSession.Value;
        _compileSession.Value = compileSession;
    }

    public static CompileSession? Current => _compileSession.Value;

    public static CompileSession GetCurrentThrowIfNull() => Current ?? throw new InvalidOperationException($"Current {nameof(CompileSession)} is not set");

    public void Dispose()
    {
        if (_initialized)
        {
            _compileSession.Value = _originalCompileSession;
        }
    }
}
