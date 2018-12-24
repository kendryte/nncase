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
using NnCase.Designer.Modules.Inspector.ViewModels;
using ReactiveUI;

namespace NnCase.Designer.Modules.Inspector.Views
{
    /// <summary>
    /// InspectorView.xaml 的交互逻辑
    /// </summary>
    public partial class InspectorView : ReactiveUserControl<InspectorViewModel>
    {
        public InspectorView()
        {
            InitializeComponent();

            this.WhenActivated(d =>
            {
                this.OneWayBind(ViewModel, vm => vm.SelectedObject.Inspectors, v => v._inspectors.ItemsSource)
                    .DisposeWith(d);
            });
        }
    }
}
