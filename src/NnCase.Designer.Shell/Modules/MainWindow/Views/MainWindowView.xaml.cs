using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Disposables;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using NnCase.Designer.Modules.MainWindow.ViewModels;
using ReactiveUI;
using Splat;

namespace NnCase.Designer.Modules.MainWindow.Views
{
    /// <summary>
    /// MainWindowView.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindowView : ReactiveWindow<MainWindowViewModel>
    {
        public MainWindowView(MainWindowViewModel viewModel)
        {
            InitializeComponent();
            ViewModel = viewModel;

            this.WhenActivated(d =>
            {
                this.OneWayBind(ViewModel, vm => vm.Shell, v => v._content.ViewModel)
                    .DisposeWith(d);
            });
        }
    }
}
