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
    private int _windowWidth = 500;

    [ObservableProperty]
    private PromptDialogLevel _dialogLevel;

    [ObservableProperty]
    private bool _isError;

    [ObservableProperty]
    private string _title = string.Empty;

    public PromptDialogViewModel(string content, PromptDialogLevel level)
    {
        DialogContent = content;
        DialogLevel = level;
        IsError = level == PromptDialogLevel.Error;
        if (IsError)
        {
            Title = "错误";
        }
        else
        {
            Title = "提示";
        }

        CloseWindowCommand = new RelayCommand<Window>(CloseWindow);
        WindowWidth = content.Length * 20 + 60;
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
