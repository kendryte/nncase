using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Nncase.Quantization;
using Nncase.Studio.Views;

namespace Nncase.Studio.ViewModels;

public partial class QuantizeViewModel : ViewModelBase
{
    [ObservableProperty] private QuantizeOptions _quantizeOptions;

    [ObservableProperty] private CalibMethod _calibMethod;

    [ObservableProperty] private QuantType _quantType;

    [ObservableProperty] private QuantType _wQuantType;

    [ObservableProperty] private ModelQuantMode _modelQuantMode;

    [ObservableProperty] private string _quantSchemePath;

    public ObservableCollection<ModelQuantMode> ModelQuantModeList { get; set; }

    public QuantizeViewModel()
    {
        ModelQuantModeList = new ObservableCollection<ModelQuantMode>(Enum.GetValues<ModelQuantMode>().Skip(1).ToList());
        ModelQuantMode = ModelQuantMode.UsePTQ;
        QuantizeOptions = new();
    }

    private DataType QuantTypeToDataType(QuantType qt)
    {
        return qt switch
        {
            QuantType.Uint8 => DataTypes.UInt8,
            QuantType.Int8 => DataTypes.Int8,
            QuantType.Int16 => DataTypes.Int16,
            _ => throw new ArgumentOutOfRangeException(nameof(qt), qt, null)
        };
    }

    public void UpdateCompileOption(CompileOptions options)
    {
        QuantizeOptions.QuantType = QuantTypeToDataType(QuantType);
        QuantizeOptions.WQuantType = QuantTypeToDataType(WQuantType);
        QuantizeOptions.ModelQuantMode = ModelQuantMode;
        QuantizeOptions.CalibrationMethod = CalibMethod;
        QuantizeOptions.QuantScheme = QuantSchemePath;
        options.QuantizeOptions = this.QuantizeOptions;
    }

    public List<string> Validate()
    {
        return new();
    }
}
