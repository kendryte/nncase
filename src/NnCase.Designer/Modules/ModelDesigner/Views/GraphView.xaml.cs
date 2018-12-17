using System;
using System.Collections.Generic;
using System.Linq;
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
using NnCase.Designer.Modules.ModelDesigner.ViewModels;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.Views
{
    /// <summary>
    /// GraphView.xaml 的交互逻辑
    /// </summary>
    public partial class GraphView : ReactiveUserControl<GraphViewModel>
    {
        public GraphView()
        {
            InitializeComponent();
        }
    }
}
