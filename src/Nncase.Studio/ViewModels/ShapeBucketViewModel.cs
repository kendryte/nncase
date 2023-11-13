using System;
using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Nncase.Studio.ViewModels;

public partial class ShapeBucketViewModel : ViewModelBase
{
    [ObservableProperty] private string _segmentCount;

    [ObservableProperty] private string _fixVarMap;

    [ObservableProperty] private string _varRangeInfo;

    public ShapeBucketViewModel(ViewModelContext context)
    {
        Context = context;
    }

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
        try
        {
            map = input.Trim().Split(",").Select(x => x.Trim().Split(":")).ToDictionary(x => x[0], x => int.Parse(x[1]));
            return true;
        }
        catch (Exception e)
        {
            map = new();
            return false;
        }
    }

    public bool TryParseRangeInfo(string input, out Dictionary<string, (int Min, int Max)> map)
    {
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
        catch (Exception e)
        {
            map = new();
            return false;
        }
    }

}
