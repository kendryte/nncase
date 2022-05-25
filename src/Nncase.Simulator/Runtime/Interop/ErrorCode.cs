// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

[StructLayout(LayoutKind.Sequential)]
internal struct ErrorCode
{
    public ErrorCode(int value)
    {
        Value = value;
    }

    public int Value { get; set; }

    public bool IsSuccess => Value >= 0;

    public void ThrowIfFailed()
    {
        if (!IsSuccess)
        {
            throw new InvalidOperationException();
        }
    }
}
