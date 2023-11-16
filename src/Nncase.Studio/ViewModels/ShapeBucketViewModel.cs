// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Nncase.Studio.ViewModels;

public partial class ShapeBucketViewModel : ViewModelBase
{
    [ObservableProperty]
    private string _segmentCount = string.Empty;

    [ObservableProperty]
    private string _fixVarMap = string.Empty;

    [ObservableProperty]
    private string _varRangeInfo = string.Empty;

    public ShapeBucketViewModel(ViewModelContext context)
    {
        Context = context;
    }

    // todo: fix var map not stop, add validator for this
    public override void UpdateContext()
    {
        var options = new ShapeBucketOptions();
        options.Enable = true;
        if (int.TryParse(SegmentCount, out var count))
        {
            options.SegmentsCount = count;
        }

        if (TryParseFixVarMap(FixVarMap, out var fixVarMap))
        {
            options.FixVarMap = fixVarMap;
        }

        if (TryParseRangeInfo(VarRangeInfo, out var rangeInfo))
        {
            options.RangeInfo = rangeInfo;
        }

        Context.CompileOption.ShapeBucketOptions = options;
    }

    public bool TryParseFixVarMap(string input, out Dictionary<string, int> map)
    {
        map = new();
        if (input == string.Empty)
        {
            return false;
        }

        try
        {
            map = input.Trim().Split(",").Select(x => x.Trim().Split(":")).ToDictionary(x => x[0], x => int.Parse(x[1]));
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    public bool TryParseRangeInfo(string input, out Dictionary<string, (int Min, int Max)> map)
    {
        map = new();
        if (input == string.Empty)
        {
            return false;
        }

        try
        {
            map = input.Trim()
                .Split(";")
                .Select(x => x.Trim().Split(":"))
                .ToDictionary(x => x[0], x =>
                {
                    var pair = x[1].Split(",");
                    return (int.Parse(pair[0]), int.Parse(pair[1]));
                });
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    public override bool IsVisible() => Context.EnableShapeBucket;
}
