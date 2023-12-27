// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.Imaging;
using Nncase.Evaluator.NN;
using Nncase.Evaluator.Tensors;
using Nncase.Hosting;

namespace Nncase.Evaluator.TIR.CPU;

/// <summary>
/// CPU module.
/// </summary>
internal class CPUModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<BinaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<MatmulEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<MemcopyEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PtrOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SramPtrEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TensorLoadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TensorStoreEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UnaryEvaluator>(reuse: Reuse.Singleton);
    }
}
