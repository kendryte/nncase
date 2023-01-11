// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Quantization;

/// <summary>
/// <see cref="ICalibrationDatasetProvider"/> that generates random inputs data.
/// </summary>
public sealed class RandomCalibrationDatasetProvider : ICalibrationDatasetProvider
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RandomCalibrationDatasetProvider"/> class.
    /// </summary>
    /// <param name="vars">Input parameters.</param>
    /// <param name="samplesCount">Samples count.</param>
    public RandomCalibrationDatasetProvider(IReadOnlyList<Var> vars, int samplesCount)
    {
        Count = samplesCount;
        Samples = Enumerable.Range(0, samplesCount).Select(i =>
        {
            var values = new Dictionary<Var, IValue>();
            foreach (var var in vars)
            {
                CompilerServices.InferenceType(var);
                var shape = var.CheckedShape.Select(d => d.IsUnknown ? 1 : d.FixedValue).ToArray();
                var value = IR.F.Random.Normal(var.CheckedDataType, 0, 1, 0, shape).Evaluate();
                values.Add(var, value);
            }

            return values;
        }).ToAsyncEnumerable();
    }

    /// <inheritdoc/>
    public int? Count { get; }

    /// <inheritdoc/>
    public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }
}
