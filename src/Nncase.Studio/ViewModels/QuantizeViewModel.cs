﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Media.Fonts;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Studio.Util;
using Nncase.Studio.Views;

namespace Nncase.Studio.ViewModels;

public partial class QuantizeViewModel : ViewModelBase
{
    [ObservableProperty]
    private QuantizeOptions _quantizeOptionsValue;

    [ObservableProperty]
    private CalibMethod _calibMethodValue;

    [ObservableProperty]
    private QuantType _quantTypeValue;

    [ObservableProperty]
    private QuantType _wQuantTypeValue;

    [ObservableProperty]
    private ModelQuantMode _modelQuantModeValue;

    [ObservableProperty]
    private bool _mixQuantize;

    [ObservableProperty]
    private bool _exportQuantScheme;

    [ObservableProperty]
    private string _quantSchemePath = string.Empty;

    [ObservableProperty]
    private string _exportQuantSchemePath = string.Empty;

    [ObservableProperty]
    private string _calibDir = string.Empty;

    private string[][] _multiInputFiles = Array.Empty<string[]>();

    public QuantizeViewModel(ViewModelContext context)
    {
        ModelQuantModeList = new ObservableCollection<ModelQuantMode>(Enum.GetValues<ModelQuantMode>().Skip(1).ToList());
        ModelQuantModeValue = ModelQuantMode.UsePTQ;
        QuantizeOptionsValue = new();
        Context = context;
    }

    public ObservableCollection<ModelQuantMode> ModelQuantModeList { get; set; }

    [RelayCommand]
    public async Task SelectQuantScheme()
    {
        var path = await Context.OpenFile(PickerOptions.JsonPickerOptions);
        if (path.Count == 0)
        {
            return;
        }

        var json = path[0];
        if (Path.GetExtension(json) != ".json")
        {
            Context.OpenDialog("QuantScheme Should use .json");
            return;
        }

        QuantSchemePath = json;
    }

    [RelayCommand]
    public async Task SelectCalibrationDataSet()
    {
        var path = await Context.OpenFolder(PickerOptions.FolderPickerOpenOptions);
        if (path == string.Empty)
        {
            return;
        }

        var inputFiles = Directory.GetFiles(path);

        try
        {
            var n = inputFiles.Where(f => Path.GetExtension(f) == ".npy").GroupBy(s => Path.GetFileName(s).Split("_")[0]);
            _multiInputFiles = n.Select(group =>
            {
                var value = group.ToArray();
                var one = value.OrderBy(s => int.Parse(Path.GetFileName(s).Split("_")[1])).ToArray();
                return one;
            }).ToArray();
        }
        catch (Exception e)
        {
            Context.OpenDialog($"文件夹中的文件解析失败，请检查文件名是否符合格式。\n{e.Message}");
            return;
        }

        if (inputFiles.Length == 0)
        {
            Context.OpenDialog("empty dir");
            return;
        }

        CalibDir = path;
    }

    [RelayCommand]
    public void ShowCalibFormat()
    {
        new QuantizeCalibWindow().Show();
    }

    public ICalibrationDatasetProvider? LoadCalibFiles()
    {
        try
        {
            var samples = _multiInputFiles.Select(files =>
            {
                var input = files.Select(DataUtil.ReadNumpyAsTensor).ToArray();
                var samples = Context.Params.Zip(input)
                    .ToDictionary(pair => pair.First, pair => (IValue)Value.FromTensor(pair.Second));
                return samples;
            }).ToArray();

            if (Context.Params.Length == 0)
            {
                Context.OpenDialog("Should Import Model first");
                return null;
            }

            return new SelfInputCalibrationDatasetProvider(samples);
        }
        catch (Exception e)
        {
            Context.OpenDialog(e.Message);
            return null;
        }
    }

    public override void UpdateViewModelCore(CompileConfig config)
    {
        QuantizeOptionsValue = config.CompileOption.QuantizeOptions;
        if (ExportQuantSchemePath == string.Empty)
        {
            ExportQuantSchemePath = Path.Join(config.CompileOption.DumpDir, "QuantScheme.json");
        }

        MixQuantize = config.MixQuantize;
    }

    public override void UpdateConfig(CompileConfig config)
    {
        QuantizeOptionsValue.CalibrationMethod = CalibMethodValue;
        QuantizeOptionsValue.QuantType = DataUtil.QuantTypeToDataType(QuantTypeValue);
        QuantizeOptionsValue.WQuantType = DataUtil.QuantTypeToDataType(WQuantTypeValue);
        QuantizeOptionsValue.ModelQuantMode = ModelQuantModeValue;
        QuantizeOptionsValue.QuantScheme = QuantSchemePath;
        QuantizeOptionsValue.ExportQuantScheme = ExportQuantScheme;
        QuantizeOptionsValue.ExportQuantSchemePath = ExportQuantSchemePath;
        config.CompileOption.QuantizeOptions = QuantizeOptionsValue;
    }

    public override List<string> CheckViewModel()
    {
        var list = new List<string>();
        if (Context.CompileConfig.MixQuantize)
        {
            if (QuantSchemePath == string.Empty)
            {
                list.Add("QuantSchemePath should not be empty");
            }
            else if (!File.Exists(QuantSchemePath))
            {
                list.Add("QuantSchemePath not exist, please check file and path");
            }
        }
        else
        {
            if (Directory.Exists(CalibDir))
            {
                if (_multiInputFiles.Length == 0)
                {
                    list.Add("CalibDir don't exist any .npy file");
                }
            }
            else
            {
                list.Add($"CalibDir {CalibDir} not exist");
            }

            if (QuantTypeValue == QuantType.Int16 && WQuantTypeValue == QuantType.Int16 &&
                string.Equals(Context.CompileConfig.Target, "k230", StringComparison.OrdinalIgnoreCase))
            {
                list.Add("k230 not support QuantType and WeightsQuantType are int16");
            }
        }

        return list;
    }

    public override bool IsVisible() => Context.CompileConfig.UseQuantize;
}

public sealed class SelfInputCalibrationDatasetProvider : ICalibrationDatasetProvider
{
    private readonly int _count = 1;

    private readonly IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> _samples;

    public SelfInputCalibrationDatasetProvider(IReadOnlyDictionary<Var, IValue>[] samples)
    {
        _samples = samples.ToAsyncEnumerable();
    }

    public int? Count => _count;

    public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples => _samples;
}
