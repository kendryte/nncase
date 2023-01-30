// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

public class MixQuantInfo
{
    public bool HasBindedMixQuantInfo { get; set; }

    public DataType MarkerQuantType { get; set; } = DataTypes.Float32;

    public List<QuantParam> QuantParameter { get; set; } = new List<QuantParam>();

    public bool DoSquant { get; set; }

    public TensorConst? U8FineTunedWeights { get; set; }

    public TensorConst? U8FineTunedWeightsRangesByChannel { get; set; }

    public TensorConst? I8FineTunedWeights { get; set; }

    public TensorConst? I8FineTunedWeightsRangesByChannel { get; set; }
}

public class AdaQuantInfo
{
    public QuantParam InputQuantParameter { get; set; } = new QuantParam(0, 1.0f);

    public Tensor? AdaRoundRefTensor { get; set; }
}

/// <summary>
/// The marker expression, it's can attach the attribute on the target.
/// </summary>
/// <param name="Name"> Name will belong to <see cref="WellknownMarkerNames"/>. </param>
/// <param name="Target"> expr target. </param>
/// <param name="Attribute"> expr attribute. </param>
public sealed record Marker(string Name, Expr Target, Expr Attribute) : Expr
{
    /// <summary>
    /// Gets or sets the mix quant info.
    /// </summary>
    public MixQuantInfo? MixQuantInfo { get; set; }

    /// <summary>
    /// Gets or sets the ada quant info.
    /// </summary>
    public AdaQuantInfo? AdaQuantInfo { get; set; }
}

/// <summary>
/// staic marker name collection.
/// </summary>
public static class WellknownMarkerNames
{
    /// <summary>
    /// attribute. <seealso cref="IR.Math.RangeOf"/>
    /// </summary>
    public static readonly string RangeOf = "RangeOf";
}
