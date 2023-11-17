// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using NetFabric.Hyperlinq;

namespace Nncase.Studio.ViewModels;

public partial class ShapeBucketViewModel : ViewModelBase
{
    [ObservableProperty]
    private int _segmentCount = 5;

    [ValidBucketVarMap]
    [ObservableProperty]
    private string _fixVarMap = string.Empty;

    [ValidBucketRangeInfo]
    [ObservableProperty]
    private string _varRangeInfo = string.Empty;

    public ShapeBucketViewModel(ViewModelContext context)
    {
        Context = context;
    }

    public override void UpdateContext()
    {
        var options = new ShapeBucketOptions();
        options.Enable = true;
        options.SegmentsCount = SegmentCount;

        if (DataUtil.TryParseFixVarMap(FixVarMap, out var fixVarMap))
        {
            options.FixVarMap = fixVarMap;
        }

        if (DataUtil.TryParseRangeInfo(VarRangeInfo, out var rangeInfo))
        {
            options.RangeInfo = rangeInfo;
        }

        Context.CompileOption.ShapeBucketOptions = options;
    }

    public override bool IsVisible() => Context.EnableShapeBucket;

    public override List<string> CheckViewModel()
    {
        if (!CustomValidator.ValidateViewModel(this, out var results))
        {
            return results.Select(x => x.ErrorMessage!).ToList();
        }

        return new();
    }
}
