// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Nncase.Quantization;
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

    public QuantizeViewModel(ViewModelContext context)
    {
        ModelQuantModeList = new ObservableCollection<ModelQuantMode>(Enum.GetValues<ModelQuantMode>().Skip(1).ToList());
        ModelQuantModeValue = ModelQuantMode.UsePTQ;
        QuantizeOptionsValue = new();
        Context = context;
    }

    public ObservableCollection<ModelQuantMode> ModelQuantModeList { get; set; }

    public void UpdateCompileOption(CompileOptions options)
    {
        QuantizeOptionsValue.QuantType = QuantTypeToDataType(QuantTypeValue);
        QuantizeOptionsValue.WQuantType = QuantTypeToDataType(WQuantTypeValue);
        QuantizeOptionsValue.ModelQuantMode = ModelQuantModeValue;
        QuantizeOptionsValue.CalibrationMethod = CalibMethodValue;
        QuantizeOptionsValue.QuantScheme = QuantSchemePath;
        QuantizeOptionsValue.ExportQuantScheme = ExportQuantScheme;
        options.QuantizeOptions = QuantizeOptionsValue;
    }

    public List<string> Validate()
    {
        return new();
    }

    private DataType QuantTypeToDataType(QuantType qt)
    {
        return qt switch
        {
            QuantType.Uint8 => DataTypes.UInt8,
            QuantType.Int8 => DataTypes.Int8,
            QuantType.Int16 => DataTypes.Int16,
            _ => throw new ArgumentOutOfRangeException(nameof(qt), qt, null),
        };
    }

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
        if (inputFiles.Length == 0)
        {
            Context.OpenDialog("empty dir");
            return;
        }

        var input = Helper.ReadInput(inputFiles).ToArray();
        if (input.Length == 0)
        {
            Context.OpenDialog("no file is loaded, only support .npy");
            return;
        }

        var samples = Context.Entry.Parameters.ToArray().Zip(input)
            .ToDictionary(pair => pair.First, pair => (IValue)Value.FromTensor(pair.Second));
        QuantizeOptionsValue.CalibrationDataset = new SelfInputCalibrationDatasetProvider(samples);
    }
}
