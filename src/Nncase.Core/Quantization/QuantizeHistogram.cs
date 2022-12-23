// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Quantization;

public class QuantizeHistogram<T>
{
    public QuantizeHistogram(List<T> srcBin, List<T> dstBin)
    {
        SrcBin = srcBin;
        DstBin = dstBin;
    }

    public List<T> SrcBin { get; set; }

    public List<T> DstBin { get; set; }
}
