// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

/// <summary>
/// Runtime interpreter.
/// </summary>
public sealed class RTInterpreter : IDisposable
{
    private readonly IntPtr _handle;
    private MemoryHandle _pinnedModelBuffer;
    private bool _disposedValue;
    private RTFunction? _entry;

    /// <summary>
    /// Initializes a new instance of the <see cref="RTInterpreter"/> class.
    /// </summary>
    public RTInterpreter()
    {
        Native.InterpCreate(out _handle).ThrowIfFailed();
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="RTInterpreter"/> class.
    /// </summary>
    ~RTInterpreter()
    {
        Dispose(disposing: false);
    }

    /// <summary>
    /// Gets entry function.
    /// </summary>
    public RTFunction? Entry
    {
        get
        {
            if (_entry == null)
            {
                Native.InterpGetEntryFunction(_handle, out var entry).ThrowIfFailed();
                _entry = entry == IntPtr.Zero ? null : new RTFunction(this, entry);
            }

            return _entry;
        }
    }

    public void SetDumpRoot(String path)
    {
        Native.InterpSetDumpRoot(_handle, path);
    }

    /// <summary>
    /// Load kmodel.
    /// </summary>
    /// <param name="modelBuffer">KModel buffer.</param>
    public unsafe void LoadModel(Memory<byte> modelBuffer)
    {
        _pinnedModelBuffer.Dispose();
        _pinnedModelBuffer = modelBuffer.Pin();
        Native.InterpLoadModel(_handle, _pinnedModelBuffer.Pointer, (uint)modelBuffer.Length, false).ThrowIfFailed();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            if (disposing)
            {
                _pinnedModelBuffer.Dispose();
            }

            Native.InterpFree(_handle);
            _disposedValue = true;
        }
    }
}
