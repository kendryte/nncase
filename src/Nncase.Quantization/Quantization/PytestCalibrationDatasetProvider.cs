// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Quantization;

/// <summary>
/// <see cref="ICalibrationDatasetProvider"/> that get the pytest generated inputs data.
/// </summary>
public sealed class PytestCalibrationDatasetProvider : ICalibrationDatasetProvider
{
    private readonly Sample[][] _sampleSets;
    private readonly string _dataset;

    /// <summary>
    /// Initializes a new instance of the <see cref="PytestCalibrationDatasetProvider"/> class.
    /// </summary>
    /// <param name="vars">Input parameters.</param>
    /// <param name="dataset">Dataset folder path.</param>
    public PytestCalibrationDatasetProvider(IReadOnlyList<Var> vars, string dataset)
    {
        Trace.Assert(Directory.Exists(dataset), "The dataset folder path must be valid!");

        _dataset = dataset;
        var sampleItems = new List<Sample>();

        // collect the samples
        foreach (var fileName in Directory.EnumerateFiles(dataset))
        {
            if (TryParseSample(fileName, out var sample))
            {
                sampleItems.Add(sample);
            }
        }

        // group by the samples
        _sampleSets = sampleItems.GroupBy(item => item.Number).Select(g => g.OrderBy(item => item.InputIndex).ToArray()).ToArray();

        Count = _sampleSets.Length;
        Samples = _sampleSets.Select(samples =>
        {
            var values = new Dictionary<Var, IValue>();

            // check the sample length equal with vars length
            Trace.Assert(samples.Length == vars.Count, "The dataset samples length not match model inputs length!");
            foreach (var (sample, var) in samples.Zip(vars))
            {
                IValue value;
                switch (var.CheckedType)
                {
                    case TensorType tensorType:
                        {
                            int[] shape = Array.Empty<int>();
                            if (tensorType.Shape.IsFixed)
                            {
                                shape = tensorType.Shape.ToValueArray();
                            }
                            else
                            {
                                shape = sample.GetShape();
                            }

                            Trace.Assert(shape.Length != 0);
                            value = Value.FromTensor(Tensor.FromBytes(tensorType.DType, File.ReadAllBytes(sample.FileName), shape));
                            break;
                        }

                    default:
                        throw new NotSupportedException();
                }

                values.Add(var, value);
            }

            return values;
        }).ToAsyncEnumerable();
    }

    /// <inheritdoc/>
    public int? Count { get; }

    /// <inheritdoc/>
    public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }

    /// <summary>
    /// try parse the sample item.
    /// </summary>
    private bool TryParseSample(string fileName, [System.Diagnostics.CodeAnalysis.MaybeNullWhen(false)] out Sample item)
    {
        var match = Regex.Match(fileName, @"^(.+?)_(\d+)_(\d+)\.bin$");
        if (match.Success)
        {
            string name = match.Groups[1].Value;
            int n = int.Parse(match.Groups[2].Value);
            int i = int.Parse(match.Groups[3].Value);
            item = new(name, n, i);
            return true;
        }

        item = null!;
        return false;
    }

    private sealed record Sample(string Name, int Number, int InputIndex)
    {
        public string FileName => $"{Name}_{Number}_{InputIndex}.bin";

        public int[] GetShape()
        {
            using var stream = File.OpenRead($"{Name}_{Number}_{InputIndex}.txt");
            using var reader = new StreamReader(stream);
            var line = reader.ReadLine();
            int[] shape = Array.Empty<int>();
            if (line is string shapeString)
            {
                string pattern = @"\d+";
                MatchCollection matches = Regex.Matches(shapeString, pattern);
                shape = matches.Select(m => int.Parse(m.Value)).ToArray();
            }

            return shape;
        }
    }
}
