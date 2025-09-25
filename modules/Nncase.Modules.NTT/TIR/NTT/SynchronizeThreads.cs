// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class SynchronizeThreads : NTTKernelOp
{
    public override bool CanFoldConstCall => false;
}
