// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Nncase.Studio.ViewModels;

public partial class NavigatorViewModel : ViewModelBase
{
    private readonly Action<ViewModelBase> _windowUpdater;

    private readonly Action<string, PromptDialogLevel> _showDialog;

    [ObservableProperty]
    private ViewModelBase? _contentViewModel;

    [ObservableProperty]
    private int _pageIndex = 0;

    [ObservableProperty]
    private int _pageMaxIndex;

    [ObservableProperty]
    private bool _isLast;

    [ObservableProperty]
    private string _pageIndexString = string.Empty;

    public NavigatorViewModel(ObservableCollection<ViewModelBase> content, Action<string, PromptDialogLevel> showDialog, Action<ViewModelBase> windowUpdater)
    {
        ContentViewModelList = content;
        _windowUpdater = windowUpdater;
        _showDialog = showDialog;
    }

    public ObservableCollection<ViewModelBase> ContentViewModelList { get; set; } = new();

    public int PageCount => ContentViewModelList.Count;

    public void SwitchToPage(ViewModelBase page)
    {
        UpdateContentViewModel(() =>
        {
            var index = ContentViewModelList.IndexOf(page);
            PageIndex = index;
        });
    }

    [RelayCommand]
    public void SwitchPrev()
    {
        UpdateContentViewModel(() =>
        {
            do
            {
                PageIndex -= 1;
            } while (!ContentViewModelList[PageIndex].IsVisible());
        });
    }

    [RelayCommand]
    public void SwitchNext()
    {
        var check = ContentViewModel!.CheckViewModel();

        // todo: show log
        if (check.Count != 0)
        {
            _showDialog("Err:\n" + string.Join("\n", check), PromptDialogLevel.Error);
            return;
        }

        UpdateContentViewModel(() =>
        {
            do
            {
                PageIndex += 1;
            } while (!ContentViewModelList[PageIndex].IsVisible());
        });
    }

    public void UpdateContentViewModel()
    {
        UpdateContentViewModel(() => { });
    }

    public void UpdateContentViewModel(Action updateIndex)
    {
        ContentViewModel?.UpdateContext();
        updateIndex();
        PageMaxIndex = ContentViewModelList.Count - 1;
        ContentViewModel = ContentViewModelList[PageIndex];
        ContentViewModel.UpdateViewModel();
        _windowUpdater(ContentViewModel);
        PageIndexString = $"{PageIndex + 1} / {PageCount}";
        IsLast = PageIndex == PageMaxIndex;
    }

    public void InsertPageAfter(ViewModelBase page, ViewModelBase pagePosition, int offset = 0)
    {
        var i = ContentViewModelList.IndexOf(page);
        if (i == -1)
        {
            var optionIndex = ContentViewModelList.IndexOf(pagePosition);

            // insert after OptionView
            ContentViewModelList.Insert(optionIndex + offset, page);
        }
    }

    public void RemovePage(ViewModelBase page)
    {
        var i = ContentViewModelList.IndexOf(page);
        if (i != -1)
        {
            ContentViewModelList.Remove(page);
        }
    }
}
