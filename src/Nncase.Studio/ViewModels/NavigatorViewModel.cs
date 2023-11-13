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

    public NavigatorViewModel(ObservableCollection<ViewModelBase> content, Action<ViewModelBase> windowUpdater)
    {
        ContentViewModelList = content;
        _windowUpdater = windowUpdater;
    }

    public ObservableCollection<ViewModelBase> ContentViewModelList { get; set; } = new();

    public int PageCount => ContentViewModelList.Count;

    [RelayCommand]
    public void SwitchPrev()
    {
        if (PageIndex != 0)
        {
            PageIndex -= 1;
        }

        UpdateContentViewModel();
    }

    [RelayCommand]
    public void SwitchNext()
    {
        if (PageIndex != PageMaxIndex)
        {
            PageIndex += 1;
        }

        UpdateContentViewModel();
    }

    public void UpdateContentViewModel()
    {
        ContentViewModel?.UpdateContext();
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
