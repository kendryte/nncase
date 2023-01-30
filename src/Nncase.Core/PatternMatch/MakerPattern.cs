// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Marker"/>.
/// </summary>
/// <param name="NameCondition">marker name condition.</param>
/// <param name="Target">Target pattern.</param>
/// <param name="Attribute">Attribute pattern.</param>
/// <param name="Name"> name. </param>
public sealed record MarkerPattern(Func<string, bool> NameCondition, Pattern Target, Pattern Attribute, string? Name) : Pattern<Marker>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MarkerPattern"/> class.
    /// </summary>
    /// <param name="marker"><see cref="Marker"/> expression.</param>
    /// <param name="name">name.</param>
    public MarkerPattern(Marker marker, string? name)
        : this(s => s == marker.Name, marker.Target, marker.Attribute, name)
    {
    }

    protected override bool MatchLeafCore(Marker expr) => NameCondition(expr.Name);
}

public static partial class Utility
{
    /// <summary>
    /// is marker.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="name_condition"> mark name condition. </param>
    /// <param name="target">target.</param>
    /// <param name="attribute">attribute.</param>
    /// <returns> MarkerPattern. </returns>
    public static MarkerPattern IsMarker(string? name, Func<string, bool> name_condition, Pattern target, Pattern attribute) => new MarkerPattern(name_condition, target, attribute, name);

    /// <summary>
    /// is maker.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="marker_name">marker name.</param>
    /// <param name="target">target.</param>
    /// <param name="attribute">attribute.</param>
    /// <returns> MarkerPattern. </returns>
    public static MarkerPattern IsMarker(string? name, string marker_name, Pattern target, Pattern attribute) => IsMarker(name, s => s == marker_name, target, attribute);

    /// <summary>
    /// is maker without name.
    /// </summary>
    /// <param name="marker_name">marker name.</param>
    /// <param name="target">target.</param>
    /// <param name="attribute">attribute.</param>
    /// <returns> MarkerPattern. </returns>
    public static MarkerPattern IsMarker(string marker_name, Pattern target, Pattern attribute) => IsMarker(null, marker_name, target, attribute);

    /// <summary>
    /// is range of maker without name.
    /// </summary>
    /// <param name="target">target.</param>
    /// <param name="attribute">attribute.</param>
    /// <returns> MarkerPattern. </returns>
    public static MarkerPattern IsRangeOfMarker(Pattern target, Pattern attribute) => IsMarker(null, WellknownMarkerNames.RangeOf, target, attribute);

    /// <summary>
    /// is range of maker with name.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="target">target.</param>
    /// <param name="attribute">attribute.</param>
    /// <returns> MarkerPattern. </returns>
    public static MarkerPattern IsRangeOfMarker(string? name, Pattern target, Pattern attribute) => IsMarker(name, WellknownMarkerNames.RangeOf, target, attribute);
}
