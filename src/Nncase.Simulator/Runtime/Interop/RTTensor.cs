// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

public class RTTensor : RTObject
{
    internal RTTensor(IntPtr handle)
        : base(handle)
    {
    }

    public static RTTensor Create(RTDataType dataType, ReadOnlySpan<uint> dims, ReadOnlySpan<uint> strides)
    {
        throw new NotImplementedException();
    }
}
