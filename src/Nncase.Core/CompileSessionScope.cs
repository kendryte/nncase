// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

internal struct CompileSessionScope : IDisposable
{
    private static readonly AsyncLocal<CompileSession?> _compileSession = new AsyncLocal<CompileSession?>();

    private readonly CompileSession? _originalCompileSession;

    public CompileSessionScope(CompileSession compileSession)
    {
        _originalCompileSession = _compileSession.Value;
        _compileSession.Value = compileSession;
    }

    public static CompileSession Current => _compileSession.Value ?? throw new InvalidOperationException($"Current {nameof(CompileSession)} is not set");

    public void Dispose()
    {
        _compileSession.Value = _originalCompileSession;
    }
}
