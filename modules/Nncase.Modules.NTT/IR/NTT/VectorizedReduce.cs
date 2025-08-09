// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NTT;

[PatternFunctionalGenerator]
public sealed partial class VectorizedReduce : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(VectorizedReduce), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo PadedNums = new(typeof(VectorizedReduce), 1, "padedNums", IsShapeType());

    public ReduceOp ReduceOp { get; }

    public IRArray<int> Axes { get; }

    public float InitValue { get; }

    public bool KeepDims { get; }

    public IRArray<int> VectorizedAxes { get; }

    public static (int[] OutVectorizeAxes, Dimension[] OutPadNums, int[] OutLanes, RankedShape OutShape) ComputeOutputInfo(VectorizedReduce target, Dimension[] inPadedNums, RankedShape inShape, int[] inLanes)
    {
        var vectorizedAxes = target.VectorizedAxes.ToList();
        var padedNums = inPadedNums.ToList();
        var lanes = inLanes.ToList();
        var shape = inShape.ToList(); // note the inshape is vectorized.
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

            if (vectorizedAxes.IndexOf(axis) is int j && j != -1)
            {
                vectorizedAxes.Remove(axis);
                padedNums.RemoveAt(j);
                lanes.RemoveAt(j);
            }

            if (!target.KeepDims)
            {
                for (int i = 0; i < vectorizedAxes.Count; i++)
                {
                    if (vectorizedAxes[i] > axis)
                    {
                        vectorizedAxes[i]--;
                    }
                }
            }
        }

        return (vectorizedAxes.ToArray(), padedNums.ToArray(), lanes.ToArray(), new(shape));
    }

    public override string DisplayProperty() => $"ReduceOp.{ReduceOp}, Axes: {{{string.Join(",", Axes)}}}, InitValue: {InitValue}, KeepDims: {KeepDims}, VectorizedAxes: {{{string.Join(",", VectorizedAxes)}}}, PadedNums: {{{string.Join(",", PadedNums)}}}";
}
