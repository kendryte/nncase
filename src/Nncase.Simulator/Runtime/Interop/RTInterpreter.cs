// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

/// <summary>
/// Runtime interpreter.
/// </summary>
public sealed class RTInterpreter : SafeHandle
{
    private MemoryHandle _pinnedModelBuffer;
    // private bool _disposedValue;
    private RTFunction? _entry;

    /// <summary>
    /// Initializes a new instance of the <see cref="RTInterpreter"/> class.
    /// </summary>
    internal RTInterpreter() : base(IntPtr.Zero, true)
    {
    }

    internal RTInterpreter(IntPtr handle) : base(handle, true)
    {
    }

    /// <summary>
    /// Create the Interpreter
    /// </summary>
    /// <returns></returns>
    public static RTInterpreter Create()
    {
        Native.InterpCreate(out var interp).ThrowIfFailed();
        return interp;
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="RTInterpreter"/> class.
    /// </summary>
    // ~RTInterpreter()
    // {
    //     Dispose(disposing: false);
    // }

    /// <summary>
    /// Gets entry function.
    /// </summary>
    public RTFunction? Entry
    {
        get
        {
            if (_entry == null)
            {
                Native.InterpGetEntryFunction(this, out var entry).ThrowIfFailed();
                _entry = entry == IntPtr.Zero ? null : new RTFunction(this, entry);
            }

            return _entry;
        }
    }

    /// <inheritdoc/>
    public override bool IsInvalid => handle == IntPtr.Zero;

    /// <summary>
    /// set the runtim dump root dir
    /// </summary>
    /// <param name="root">root dir.</param>
    public void SetDumpRoot(string root)
    {
        if (!Directory.Exists(root))
            Directory.CreateDirectory(root);
        Native.InterpSetDumpRoot(this, root);
    }

    /// <summary>
    /// Load kmodel.
    /// </summary>
    /// <param name="modelBuffer">KModel buffer.</param>
    public unsafe void LoadModel(Memory<byte> modelBuffer)
    {
        _pinnedModelBuffer.Dispose();
        _pinnedModelBuffer = modelBuffer.Pin();
        Native.InterpLoadModel(this, _pinnedModelBuffer.Pointer, (uint)modelBuffer.Length, false).ThrowIfFailed();
    }

    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        _pinnedModelBuffer.Dispose();
        return Native.InterpFree(handle).IsSuccess;
    }

    /// <inheritdoc/>
    // public void Dispose()
    // {
    //     Dispose(disposing: true);
    //     GC.SuppressFinalize(this);
    // }

    // private void Dispose(bool disposing)
    // {
    //     if (!_disposedValue)
    //     {
    //         if (disposing)
    //         {
    //             _pinnedModelBuffer.Dispose();
    //         }

    //         Native.InterpFree(_handle);
    //         _disposedValue = true;
    //     }
    // }
}
