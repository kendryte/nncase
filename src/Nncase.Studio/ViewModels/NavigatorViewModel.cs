using System;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Nncase.Studio.ViewModels;

public partial class NavigatorViewModel : ViewModelBase
{
    [ObservableProperty]
    private ViewModelBase _contentViewModel;

    public int PageCount => ContentViewModelList.Count;

    [ObservableProperty]
    private int _pageIndex = 0;

    [ObservableProperty]
    private int _pageMaxIndex;

    [ObservableProperty]
    private bool _isLast;

    [ObservableProperty]
    private string _pageIndexString;

    private Action<ViewModelBase> _windowUpdater;

    public ObservableCollection<ViewModelBase> ContentViewModelList { get; set; } = new();

    public NavigatorViewModel(ObservableCollection<ViewModelBase> content, Action<ViewModelBase> windowUpdater)
    {
        ContentViewModelList = content;
        _windowUpdater = windowUpdater;
    }

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
        PageMaxIndex = ContentViewModelList.Count - 1;
        ContentViewModel = ContentViewModelList[PageIndex];
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
