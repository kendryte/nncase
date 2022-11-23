﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;
public class MixQuantInfo
{
    public bool HasBindedMixQuantInfo = false;
    public DataType MarkerQuantType = DataTypes.Float32;
    public List<QuantParam> QuantParameter = new List<QuantParam>();
    public bool DoSquant = false;
    public TensorConst U8FineTunedWeights;
    public TensorConst U8FineTunedWeightsRangesByChannel;
    public TensorConst I8FineTunedWeights;
    public TensorConst I8FineTunedWeightsRangesByChannel;
}

/// <summary>
/// The marker expression, it's can attach the attribute on the target.
/// </summary>
/// <param name="Name"> Name will belong to <see cref="WellknownMarkerNames"/> </param>
/// <param name="Target"> expr target. </param>
/// <param name="Attribute"> expr attribute. </param>
public sealed record Marker(string Name, Expr Target, Expr Attribute) : Expr
{
    /// <summary>
    /// Gets Target.
    /// </summary>
    public Expr MarkerTarget()
    {
        return Target;
    }

    public MixQuantInfo mixQuantInfo;
}

/// <summary>
/// staic marker name collection
/// </summary>
public static class WellknownMarkerNames
{
    /// <summary>
    /// attribute <seealso cref="IR.Math.RangeOf"/>
    /// </summary>
    public static readonly string RangeOf = "RangeOf";
}
