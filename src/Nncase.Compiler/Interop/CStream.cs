// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Compiler.Interop;

/// <summary>
/// C Stream method table.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct CStreamMT
{
    public delegate* unmanaged<IntPtr, void> AddRefPtr;
    public delegate* unmanaged<IntPtr, void> ReleasePtr;
    public delegate* unmanaged<IntPtr, int> CanReadPtr;
    public delegate* unmanaged<IntPtr, int> CanSeekPtr;
    public delegate* unmanaged<IntPtr, int> CanWritePtr;
    public delegate* unmanaged<IntPtr, void> FlushPtr;
    public delegate* unmanaged<IntPtr, long> GetLengthPtr;
    public delegate* unmanaged<IntPtr, long, void> SetLengthPtr;
    public delegate* unmanaged<IntPtr, long> GetPositionPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, nuint> ReadPtr;
    public delegate* unmanaged<IntPtr, long, SeekOrigin, long> SeekPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> WritePtr;
}

internal unsafe class CStream : Stream
{
    private readonly CStreamMT* _mt;
    private IntPtr _handle;

    public CStream(CStreamMT* mt, IntPtr handle)
    {
        _mt = mt;
        _handle = handle;
        _mt->AddRefPtr(_handle);
    }

    ~CStream()
    {
        Dispose(false);
    }

    public override bool CanRead => _mt->CanReadPtr(_handle) != 0;

    public override bool CanSeek => _mt->CanSeekPtr(_handle) != 0;

    public override bool CanWrite => _mt->CanWritePtr(_handle) != 0;

    public override long Length => _mt->GetLengthPtr(_handle);

    public override long Position
    {
        get => _mt->GetPositionPtr(_handle);
        set
        {
            if (_mt->SeekPtr(_handle, value, SeekOrigin.Begin) != value)
            {
                SetLength(value);
                if (_mt->SeekPtr(_handle, value, SeekOrigin.Begin) != value)
                {
                    throw new InvalidOperationException("Seek failed.");
                }
            }
        }
    }

    public override void Flush() =>
        _mt->FlushPtr(_handle);

    public override int Read(byte[] buffer, int offset, int count)
    {
        fixed (byte* ptr = buffer)
        {
            return (int)_mt->ReadPtr(_handle, ptr + offset, (nuint)count);
        }
    }

    public override long Seek(long offset, SeekOrigin origin) =>
        _mt->SeekPtr(_handle, offset, origin);

    public override void SetLength(long value) =>
        _mt->SetLengthPtr(_handle, value);

    public override void Write(byte[] buffer, int offset, int count)
    {
        fixed (byte* ptr = buffer)
        {
            _mt->WritePtr(_handle, ptr + offset, (nuint)count);
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (_handle != IntPtr.Zero)
        {
            _mt->ReleasePtr(_handle);
            _handle = IntPtr.Zero;
        }

        base.Dispose(disposing);
    }
}
