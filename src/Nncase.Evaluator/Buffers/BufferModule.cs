// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.Tensors;
using Nncase.Hosting;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Buffer module.
/// </summary>
internal class BufferModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<DDrOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<BufferIndexOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<BaseMentOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<StrideOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<AllocateEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UninitializedEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<BufferLoadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<BufferStoreEvaluator>(reuse: Reuse.Singleton);
    }
}
