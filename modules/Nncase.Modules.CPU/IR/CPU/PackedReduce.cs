// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.CPU;

[PatternFunctionalGenerator]
public sealed partial class PackedReduce : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(PackedReduce), 0, "input", ParameterKind.Input);

    public ReduceOp ReduceOp { get; }

    public IRArray<int> Axes { get; }

    public float InitValue { get; }

    public bool KeepDims { get; }

    public IRArray<int> PackedAxes { get; }

    public IRArray<int> PadedNums { get; }

    public static (int[] OutPackAxes, int[] OutPadNums, int[] OutLanes, int[] OutShape) ComputeOutputInfo(PackedReduce target, int[] inShape, int[] inLanes)
    {
        var packedAxes = target.PackedAxes.ToList();
        var padedNums = target.PadedNums.ToList();
        var lanes = inLanes.ToList();
        var shape = inShape.ToList(); // note the inshape is packed.
        var offset = 0;
        foreach (var axis in target.Axes)
        {
            if (target.KeepDims)
            {
                shape[axis] = 1;
            }
            else
            {
                shape.RemoveAt(offset + axis);
                offset--;
            }

            if (packedAxes.IndexOf(axis) is int j && j != -1)
            {
                packedAxes.Remove(axis);
                padedNums.RemoveAt(j);
                lanes.RemoveAt(j);
                for (int i = 0; i < packedAxes.Count; i++)
                {
                    if (packedAxes[i] > axis)
                    {
                        packedAxes[i]--;
                    }
                }
            }
        }

        return (packedAxes.ToArray(), padedNums.ToArray(), lanes.ToArray(), shape.ToArray());
    }

    public override string DisplayProperty() => $"ReduceOp.{ReduceOp}, Axes: {{{string.Join(",", Axes)}}}, InitValue: {InitValue}, KeepDims: {KeepDims}, PackedAxes: {{{string.Join(",", PackedAxes)}}}, PadedNums: {{{string.Join(",", PadedNums)}}}";
}
