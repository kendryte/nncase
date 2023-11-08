// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Nncase.Studio.ViewModels;

public partial class PreprocessViewModel : ViewModelBase
{
    [ObservableProperty]
    [ValidIntArray]
    private string _inputShape = string.Empty;

    [ObservableProperty]
    private InputType _inputTypeValue;

    [ObservableProperty]
    private string _inputTypeString = string.Empty;

    public PreprocessViewModel(ViewModelContext context)
    {
        this.Context = context;
        RangeMax = "1";
        RangeMin = "-1";
        Mean = "1";
        Std = "1";
        LetterBoxValue = "0";
        InputShape = "1, 3, 24, 24";
        InputTypeValue = InputType.Float32;
    }

    public string LayoutWatermark { get; set; } = "e.g. NHWC or NCHW or 0,2,3,1";

    public string ShapeWaterMark { get; set; } = "e.g. 1,3,224,224";

    public string ListNumberWaterMark { get; set; } = "e.g. 2.9,3,5";

    [ValidLayout]
    public string InputLayout { get; set; } = "NCHW";

    [ValidLayout]
    public string OutputLayout { get; set; } = "NCHW";

    [ValidLayout]
    public string ModelLayout { get; set; } = "NCHW";

    [ValidFloatArray]
    public bool SwapRB { get; set; }

    [ValidFloat]
    public string RangeMin { get; set; }

    [ValidFloat]
    public string RangeMax { get; set; }

    [ValidFloat]
    public string LetterBoxValue { get; set; }

    [ValidFloatArray]
    public string Mean { get; set; }

    [ValidFloatArray]
    public string Std { get; set; }

    public override void UpdateContext()
    {
        var mean = Mean.Split(",").Select(float.Parse).ToArray();
        var std = Std.Split(",").Select(float.Parse).ToArray();
        var inShape = Std.Split(",").Select(int.Parse).ToArray();
        var rangeMin = float.Parse(RangeMin);
        var rangeMax = float.Parse(RangeMax);

        // todo: check min and max
        var letterBoxValue = float.Parse(LetterBoxValue);

        // todo: mean std LetterBoxValue，全部by tensor或者部分by channel
        Context.CompileOption.Mean = mean;
        Context.CompileOption.Std = std;
        Context.CompileOption.InputShape = inShape;
        Context.CompileOption.InputRange = new[] { rangeMin, rangeMax };
        Context.CompileOption.InputType = InputTypeValue;
        Context.CompileOption.LetterBoxValue = letterBoxValue;
    }

    public override List<string> CheckViewModel()
    {
        var rangeMin = float.Parse(RangeMin);
        var rangeMax = float.Parse(RangeMax);
        if (rangeMax < rangeMin)
        {
            var l = new List<string>();
            l.Add("Invalid Range");
            return l;
        }

        return new();
    }
}
