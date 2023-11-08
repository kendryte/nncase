// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Windows.Input;
using Avalonia.Controls;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Nncase.Studio.ViewModels;

public partial class PromptDialogViewModel : ViewModelBase
{
    [ObservableProperty]
    private string _dialogContent = string.Empty;

    [ObservableProperty]
    private PromptDialogLevel _dialogLevel;

    public PromptDialogViewModel()
    {
        CloseWindowCommand = new RelayCommand<Window>(CloseWindow);
    }

    public RelayCommand<Window> CloseWindowCommand { get; private set; }

    private void CloseWindow(Window? window)
    {
        if (window != null)
        {
            window.Close();
        }
    }
}
