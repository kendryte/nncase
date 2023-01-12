﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;

namespace Nncase.Diagnostics;

/// <summary>
/// <see cref="IDumpper"/> scope.
/// </summary>
public struct DumpScope : IDisposable
{
    private static readonly AsyncLocal<IDumpper> _dumpper = new AsyncLocal<IDumpper>();

    private readonly bool _initialized;
    private readonly IDumpper _originalDumpper;

    /// <summary>
    /// Initializes a new instance of the <see cref="DumpScope"/> struct.
    /// </summary>
    /// <param name="subDirectory">Sub directory.</param>
    /// <param name="serviceProvider">Service provider.</param>
    public DumpScope(string subDirectory, IServiceProvider? serviceProvider = null)
    {
        _initialized = true;
        _originalDumpper = GetCurrent(serviceProvider);
        _dumpper.Value = _originalDumpper.CreateSubDummper(subDirectory);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DumpScope"/> struct.
    /// </summary>
    /// <param name="newDumpper">New dumpper.</param>
    /// <param name="serviceProvider">Service provider.</param>
    public DumpScope(IDumpper newDumpper, IServiceProvider? serviceProvider = null)
    {
        _initialized = true;
        _originalDumpper = GetCurrent(serviceProvider);
        _dumpper.Value = newDumpper;
    }

    /// <summary>
    /// Gets current dumpper.
    /// </summary>
    public static IDumpper Current => GetCurrent(null);

    /// <summary>
    /// Gets current <see cref="IDumpper"/> or use root of scope.
    /// </summary>
    /// <param name="serviceProvider">Service provider.</param>
    /// <returns>Current dumpper.</returns>
    public static IDumpper GetCurrent(IServiceProvider? serviceProvider) =>
        _dumpper.Value ??= (serviceProvider ?? CompileSessionScope.Current)?.GetRequiredService<IDumpperFactory>().Root ?? NullDumpper.Instance;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_initialized)
        {
            _dumpper.Value = _originalDumpper;
        }
    }
}
