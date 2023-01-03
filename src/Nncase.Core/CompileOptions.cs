// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Diagnostics;
using Nncase.Quantization;

namespace Nncase;

/// <summary>
/// Compile options.
/// </summary>
public sealed record CompileOptions(string InputFile, string InputFormat, DumpFlags DumpFlags, string DumpDir, QuantizeOptions QuantizeOptions)
{
}
