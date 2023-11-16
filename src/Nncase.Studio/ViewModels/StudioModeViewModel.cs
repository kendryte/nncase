// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using CommunityToolkit.Mvvm.Input;

namespace Nncase.Studio.ViewModels;

public partial class StudioModeViewModel : ViewModelBase
{
    public StudioModeViewModel(ViewModelContext context)
    {
        Context = context;
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
