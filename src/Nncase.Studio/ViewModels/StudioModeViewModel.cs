// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.Input;
using Newtonsoft.Json;
using Nncase.Studio.Util;

namespace Nncase.Studio.ViewModels;

public partial class StudioModeViewModel : ViewModelBase
{
    public StudioModeViewModel(ViewModelContext context)
    {
        Context = context;
    }

    [RelayCommand]
    public async Task ImportStudioConfig()
    {
        var selectFiles = await Context.OpenFile(PickerOptions.JsonPickerOptions);
        if (selectFiles.Count == 0)
        {
            return;
        }
        else
        {
            var configFile = selectFiles[0];
            var conf = JsonConvert.DeserializeObject<CompileConfig>(File.ReadAllText(configFile));
            Context.ImportConfig(conf!);
        }

        SwitchToImportView();
    }

    [RelayCommand]
    private void SwitchToImportView()
    {
        Context.SwitchToPage(typeof(ImportViewModel));
    }

    [RelayCommand]
    private void SwitchToSimulateView()
    {
        Context.SwitchToPage(typeof(SimulateViewModel));
    }
}
