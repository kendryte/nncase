// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.XPU;

public sealed partial class Concat : XPUKernelOp
{
    public int Axis { get; }
}
