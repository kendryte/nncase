// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

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
public sealed class Marker : Expr, IEquatable<Marker?>
{
    private readonly string _name;

    /// <summary>
    /// Initializes a new instance of the <see cref="Marker"/> class.
    /// </summary>
    /// <param name="name">Name will belong to <see cref="WellknownMarkerNames"/>.</param>
    /// <param name="target">expr target.</param>
    /// <param name="attribute">expr attribute.</param>
    public Marker(string name, Expr target, Expr attribute)
        : base(new[] { target, attribute })
    {
        _name = name;
    }

    public string Name => _name;

    public Expr Target => Operands[0];

    public Expr Attribute => Operands[1];

    /// <summary>
    /// Gets or sets the mix quant info.
    /// </summary>
    public MixQuantInfo? MixQuantInfo { get; set; }

    /// <summary>
    /// Gets or sets the ada quant info.
    /// </summary>
    public AdaQuantInfo? AdaQuantInfo { get; set; }

    public static bool operator ==(Marker? left, Marker? right) => EqualityComparer<Marker>.Default.Equals(left, right);

    public static bool operator !=(Marker? left, Marker? right) => !(left == right);

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitMarker(this, context);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as Marker);

    /// <inheritdoc/>
    public bool Equals(Marker? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && base.Equals(other) && Name == other.Name;
    }

    public Marker With(string? name = null, Expr? target = null, Expr? attribute = null, MixQuantInfo? mixQuantInfo = null, AdaQuantInfo? adaQuantInfo = null)
        => new Marker(name ?? Name, target ?? Target, attribute ?? Attribute)
        {
            MixQuantInfo = mixQuantInfo ?? MixQuantInfo,
            AdaQuantInfo = adaQuantInfo ?? AdaQuantInfo,
        };

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(base.GetHashCodeCore(), Name);
}
