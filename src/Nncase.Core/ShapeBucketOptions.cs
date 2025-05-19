// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase
{
    public record ShapeBucketOptions
    {
        public bool Enable { get; set; }

        public Dictionary<Var, Dimension[]> VarMap { get; set; } = new();

        public Dictionary<string, (int Min, int Max)> RangeInfo { get; set; } = new();

        public int SegmentsCount { get; set; }

        public Dictionary<string, int> FixVarMap { get; set; } = new();

        public static ShapeBucketOptions Default => new();

        public static ShapeBucketOptions CloneFrom(ShapeBucketOptions from)
        {
            var options = new ShapeBucketOptions();
            options.Enable = from.Enable;
            foreach (var (k, v) in from.VarMap)
            {
                options.VarMap.Add(k, v);
            }

            foreach (var (k, v) in from.RangeInfo)
            {
                options.RangeInfo.Add(k, v);
            }

            options.SegmentsCount = from.SegmentsCount;
            foreach (var (k, v) in from.FixVarMap)
            {
                options.FixVarMap.Add(k, v);
            }

            return options;
        }

        public static ShapeBucketOptions Create(bool enable, Dictionary<Var, Dimension[]> varMap, Dictionary<string, (int Min, int Max)> rangeInfo, int segmentsCount, Dictionary<string, int> fixVarMap)
        {
            var options = new ShapeBucketOptions();
            options.Enable = enable;
            options.VarMap = varMap;
            options.RangeInfo = rangeInfo;
            options.SegmentsCount = segmentsCount;
            options.FixVarMap = fixVarMap;
            return options;
        }
    }
}
