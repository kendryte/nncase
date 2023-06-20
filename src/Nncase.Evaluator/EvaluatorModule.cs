// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator;

/// <summary>
/// Evaluator module.
/// </summary>
internal class EvaluatorModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.Register<ITypeInferenceProvider, TypeInferenceProvider>(reuse: Reuse.Singleton);
        registrator.Register<IEvaluateProvider, EvaluateProvider>(reuse: Reuse.Singleton);
        registrator.Register<ICostEvaluateProvider, CostEvaluateProvider>(reuse: Reuse.Singleton);
        registrator.Register<IShapeEvaluateProvider, ShapeEvaluateProvider>(reuse: Reuse.Singleton);
    }
}
