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
    private static readonly AsyncLocal<CompileSession?> _externalCompileSession = new AsyncLocal<CompileSession?>();
    private static readonly AsyncLocal<bool> _refeshedExternal = new AsyncLocal<bool>() { Value = false };

    private readonly bool _initialized;
    private readonly CompileSession? _originalCompileSession;

    public CompileSessionScope(CompileSession compileSession)
    {
        _initialized = true;
        _originalCompileSession = _compileSession.Value;
        _compileSession.Value = compileSession;
    }

    public static CompileSession? Current => _compileSession.Value ?? _externalCompileSession.Value;

    /// <summary>
    /// Gets a value indicating whether gets the external compile session is referhed status.
    /// </summary>
    public static bool IsRefeshedExternal
    {
        get
        {
            bool ret = false;
            if (_refeshedExternal.Value)
            {
                ret = true;
                _refeshedExternal.Value = false;
            }

            return ret;
        }
    }

    public static CompileSession GetCurrentThrowIfNull() => Current ?? throw new InvalidOperationException($"Current {nameof(CompileSession)} is not set");

    /// <summary>
    /// refesh external CompileSession.
    /// note only call it in cApi.
    /// </summary>
    /// <param name="compileSession">external created compile session.</param>
    public static void RefeshExternal(CompileSession compileSession)
    {
        _externalCompileSession.Value = compileSession;
        _refeshedExternal.Value = true;
    }

    public void Dispose()
    {
        if (_initialized)
        {
            _compileSession.Value = _originalCompileSession;
        }
    }
}
