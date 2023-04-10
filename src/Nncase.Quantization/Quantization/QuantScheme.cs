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

internal partial class Output
{
    public string Name { get; set; } = default!;

    [JsonProperty(PropertyName = "data_type")]
    public string DataType { get; set; } = default!;

    [JsonProperty(PropertyName = "data_range")]
    public ValueRange<float>[] DataRange { get; set; } = default!;

    [JsonProperty(PropertyName = "data_range_mode")]
    public string DataRangeMode { get; set; } = default!;
}

internal partial class QuantScheme
{
    public string Version { get; set; } = default!;

    public string Model { get; set; } = default!;

    public Output[] Outputs { get; set; } = default!;
}
