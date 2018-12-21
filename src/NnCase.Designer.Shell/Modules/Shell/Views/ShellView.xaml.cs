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
using NnCase.Designer.Modules.Shell.ViewModels;
using ReactiveUI;

namespace NnCase.Designer.Modules.Shell.Views
{
    /// <summary>
    /// ShellView.xaml 的交互逻辑
    /// </summary>
    public partial class ShellView : ReactiveUserControl<ShellViewModel>
    {
        public ShellView()
        {
            InitializeComponent();

            this.WhenActivated(d =>
            {
                this.OneWayBind(ViewModel, vm => vm.MainMenu, v => v._mainMenu.ViewModel)
                    .DisposeWith(d);
                this.OneWayBind(ViewModel, vm => vm.Documents, v => v._dockingManager.DocumentsSource)
                    .DisposeWith(d);
                this.OneWayBind(ViewModel, vm => vm.Tools, v => v._dockingManager.AnchorablesSource)
                    .DisposeWith(d);
                this.Bind(ViewModel, vm => vm.ActiveLayoutItem, v => v._dockingManager.ActiveContent)
                    .DisposeWith(d);
            });
        }
    }
}
