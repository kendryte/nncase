// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.CPU;

public sealed class TDMALoad : Op
{
    public static readonly ParameterInfo Input = new(typeof(TDMALoad), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(TDMALoad), 1, "output");
}
