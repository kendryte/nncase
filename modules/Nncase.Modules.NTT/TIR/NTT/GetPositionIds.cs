// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class GetPositionIds : NTTKernelOp
{
    public static readonly ParameterInfo KVCache = new(typeof(GetPositionIds), 0, "kvcache");
    public static readonly ParameterInfo Output = new(typeof(GetPositionIds), 1, "output");
}
