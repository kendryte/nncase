// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Transform;

namespace Nncase.Quantization;

/// <summary>
/// the quantization egraph pass 
/// </summary>
public class EGraphPassWithQuantize : EGraphPass
{
    private readonly QuantizeOptions _quantizeOptions;

    /// <summary>
    /// Initializes a new instance of the <see cref="EGraphPassWithQuantize"/> class.
    /// </summary>
    /// <param name="name">Pass name.</param>
    /// <param name="quantizeOptions">options.</param>
    public EGraphPassWithQuantize(string name, QuantizeOptions quantizeOptions)
        : base(name)
    {
        _quantizeOptions = quantizeOptions;
    }

    /// <summary>
    /// the callback on the rewirte finish.
    /// 
    /// </summary>
    /// <param name="graph"></param>
    /// <param name="options"></param>
    /// <returns></returns>
    protected override async Task OnPostRewrite(EGraph graph, RunPassOptions options)
    {
        var quantizer = new Quantizer(graph);
        await quantizer.RunAsync(_quantizeOptions);
    }
}
