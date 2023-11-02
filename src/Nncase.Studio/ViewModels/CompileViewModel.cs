using System;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using Nncase.Studio.Views;
using ReactiveUI;

namespace Nncase.Studio.ViewModels;

public partial class CompileViewModel : ViewModelBase
{
    [ObservableProperty] private string _kmodelPath = "test.kmodel";
}
