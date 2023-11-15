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
using Nncase.IR;
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

    [ObservableProperty]
    private string _calibDir = string.Empty;

    private string[] _inputFiles = Array.Empty<string>();

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
        if (inputFiles.Length == 0)
        {
            Context.OpenDialog("empty dir");
            return;
        }

        CalibDir = path;
        _inputFiles = inputFiles;
    }

    public ICalibrationDatasetProvider? LoadCalibFiles()
    {
        Tensor[] input;
        try
        {
            input = DataUtil.ReadInput(_inputFiles).ToArray();
        }
        catch (Exception e)
        {
            Context.OpenDialog(e.Message);
            return null;
        }

        if (input.Length == 0)
        {
            Context.OpenDialog("no file is loaded, only support .npy");
            return null;
        }

        if (Context.Entry == null)
        {
            Context.OpenDialog("Should Import Model first");
            return null;
        }

        var samples = Context.Entry!.Parameters.ToArray().Zip(input)
            .ToDictionary(pair => pair.First, pair => (IValue)Value.FromTensor(pair.Second));
        return new SelfInputCalibrationDatasetProvider(samples);
    }

    public override void UpdateViewModel()
    {
        if (ExportQuantSchemePath == string.Empty)
        {
            ExportQuantSchemePath = Path.Join(Context.CompileOption.DumpDir, "QuantScheme.json");
        }

        MixQuantize = Context.MixQuantize;
    }

    public override void UpdateContext()
    {
        // todo: load calib data set when compile start
        QuantizeOptionsValue.CalibrationMethod = CalibMethodValue;
        QuantizeOptionsValue.QuantType = DataUtil.QuantTypeToDataType(QuantTypeValue);
        QuantizeOptionsValue.WQuantType = DataUtil.QuantTypeToDataType(WQuantTypeValue);
        QuantizeOptionsValue.ModelQuantMode = ModelQuantModeValue;
        QuantizeOptionsValue.QuantScheme = QuantSchemePath;
        QuantizeOptionsValue.ExportQuantScheme = ExportQuantScheme;
        QuantizeOptionsValue.ExportQuantSchemePath = ExportQuantSchemePath;
        Context.CompileOption.QuantizeOptions = QuantizeOptionsValue;
    }

    public override List<string> CheckViewModel()
    {
        return new();
    }

    public override bool IsVisible() => Context.UseQuantize;
}

public sealed class SelfInputCalibrationDatasetProvider : ICalibrationDatasetProvider
{
    private readonly int _count = 1;

    private readonly IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> _samples;

    public SelfInputCalibrationDatasetProvider(IReadOnlyDictionary<Var, IValue> sample)
    {
        _samples = new[] { sample }.ToAsyncEnumerable();
    }

    public int? Count => _count;

    public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples => _samples;
}
