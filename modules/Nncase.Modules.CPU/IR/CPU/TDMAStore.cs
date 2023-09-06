// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase;

public sealed class TDMAStore : Op
{
    public static readonly ParameterInfo Input = new(typeof(TDMAStore), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(TDMAStore), 1, "output");
}
