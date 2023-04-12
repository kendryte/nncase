// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Math module.
/// </summary>
internal class MathModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<BinaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ClampEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<CompareEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<CumSumEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<DequantizeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<FakeDequantizeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<FakeQuantizeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<MatMulEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<QuantizeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<QuantParamOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<RangeOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ReduceEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ReduceArgEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UnaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SelectEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ConditionEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<RequireEvaluator>(reuse: Reuse.Singleton);
    }
}
