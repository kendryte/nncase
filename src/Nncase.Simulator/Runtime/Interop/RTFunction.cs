// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

/// <summary>
/// Runtime function.
/// </summary>
public sealed class RTFunction
{
    private readonly RTInterpreter _interp;
    private readonly IntPtr _handle;

    internal RTFunction(RTInterpreter interp, IntPtr handle)
    {
        _interp = interp;
        _handle = handle;
    }

    /// <summary>
    /// Gets count of params.
    /// </summary>
    public uint ParamsCount
    {
        get
        {
            Native.FuncGetParamsSize(_handle, out var size).ThrowIfFailed();
            return size;
        }
    }

    /// <summary>
    /// Invoke function.
    /// </summary>
    /// <param name="parameters">Parameter values.</param>
    /// <returns>Result value.</returns>
    public unsafe RTValue Invoke(params RTValue[] parameters)
    {
        var paramsHandles = parameters.Select(x => x.DangerousGetHandle()).ToArray();
        fixed (IntPtr* paramsHandlesPtr = paramsHandles)
        {
            Native.FuncInvoke(this, paramsHandlesPtr, (uint)parameters.Length, out var result).ThrowIfFailed();
            GC.KeepAlive(_interp);
            return RTValue.FromHandle(result);
        }
    }
}
