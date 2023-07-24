// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Quantization;

public record class Output
{
    public string? Name { get; set; }

    public string? DataType { get; set; }

    public ValueRange<float>[]? DataRange { get; set; }

    public string? DataRangeMode { get; set; }
}

public record class QuantScheme
{
    public string? Version { get; set; }

    public string? Model { get; set; }

    public Output[]? Outputs { get; set; }
}
