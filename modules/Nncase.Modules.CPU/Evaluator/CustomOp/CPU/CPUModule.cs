// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.TIR.CPU;
using Nncase.Hosting;

namespace Nncase.Evaluator.CustomCPU;

/// <summary>
/// CPU module.
/// </summary>
internal class CPUModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<UnaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<MatMulEvaluator>(reuse: Reuse.Singleton);
    }
}
