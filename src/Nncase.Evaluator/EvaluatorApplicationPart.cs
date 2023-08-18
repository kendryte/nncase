// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Nncase.Evaluator;
using Nncase.Evaluator.Buffers;
using Nncase.Evaluator.Imaging;
using Nncase.Evaluator.Math;
using Nncase.Evaluator.NN;
using Nncase.Evaluator.Random;
using Nncase.Evaluator.RNN;
using Nncase.Evaluator.ShapeExpr;
using Nncase.Evaluator.TIR;
using Nncase.Hosting;

namespace Nncase;

/// <summary>
/// Evaluator application part extensions.
/// </summary>
public static class EvaluatorApplicationPart
{
    /// <summary>
    /// Add evaluator assembly.
    /// </summary>
    /// <param name="registrator">Service registrator.</param>
    /// <returns>Configured service registrator.</returns>
    public static IRegistrator AddEvaluator(this IRegistrator registrator)
    {
        return registrator.RegisterModule<EvaluatorModule>()
            .RegisterModule<BufferModule>()
            .RegisterModule<ImagingModule>()
            .RegisterModule<MathModule>()
            .RegisterModule<NNModule>()
            .RegisterModule<RandomModule>()
            .RegisterModule<RNNModule>()
            .RegisterModule<TensorsModule>()
            .RegisterModule<ShapeExprModule>()
            .RegisterModule<TIRModule>();
    }
}
