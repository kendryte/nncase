﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Nncase.Studio.Util;
using Nncase.Studio.Views;

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
        Context = context;
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

    [RelayCommand]
    public void ShowPreprocessOrder()
    {
        new PreprocessWindow().Show();
    }

    public override void UpdateConfig(CompileConfig config)
    {
        var mean = Mean.Split(",").Select(float.Parse).ToArray();
        var std = Std.Split(",").Select(float.Parse).ToArray();
        var inShape = Std.Split(",").Select(int.Parse).ToArray();
        var rangeMin = float.Parse(RangeMin);
        var rangeMax = float.Parse(RangeMax);
        var letterBoxValue = float.Parse(LetterBoxValue);

        config.CompileOption.InputLayout = InputLayout;
        config.CompileOption.OutputLayout = OutputLayout;
        config.CompileOption.InputType = InputTypeValue;
        config.CompileOption.InputShape = inShape;
        config.CompileOption.InputRange = new[] { rangeMin, rangeMax };
        config.CompileOption.Mean = mean;
        config.CompileOption.Std = std;
        config.CompileOption.SwapRB = SwapRB;
        config.CompileOption.ModelLayout = ModelLayout;
        config.CompileOption.LetterBoxValue = letterBoxValue;
    }

    public override void UpdateViewModelCore(CompileConfig config)
    {
        InputLayout = config.CompileOption.InputLayout;
        OutputLayout = config.CompileOption.OutputLayout;
        InputTypeValue = config.CompileOption.InputType;
        InputShape = string.Join(",", config.CompileOption.InputShape);
        var range = config.CompileOption.InputRange;
        if (range.Length == 2)
        {
            RangeMin = range[0].ToString();
            RangeMax = range[1].ToString();
        }
        else
        {
            RangeMin = string.Empty;
            RangeMax = string.Empty;
        }

        Mean = string.Join(",", config.CompileOption.Mean);
        Std = string.Join(",", config.CompileOption.Std);
        SwapRB = config.CompileOption.SwapRB;
        ModelLayout = config.CompileOption.ModelLayout;
        LetterBoxValue = config.CompileOption.LetterBoxValue.ToString();
    }

    public override List<string> CheckViewModel()
    {
        var l = new List<string>();
        if (!CustomValidator.ValidateViewModel(this, out var results))
        {
            l = l.Concat(results.Select(x => x.ErrorMessage!)).ToList();
        }

        if (float.TryParse(RangeMin, out var min) && float.TryParse(RangeMax, out var max))
        {
            if (max < min)
            {
                l.Add("Invalid Range");
            }
        }

        return l;
    }

    public override bool IsVisible()
    {
        return Context.CompileConfig.CompileOption.PreProcess && Context.CustomPreprocessMode;
    }
}
