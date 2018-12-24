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
using ReactiveUI;

namespace NnCase.Designer.Modules.Inspector.Inspectors
{
    /// <summary>
    /// TextBoxEditorView.xaml 的交互逻辑
    /// </summary>
    public abstract partial class TextBoxEditorView : UserControl
    {
        public TextBoxEditorView()
        {
            InitializeComponent();
        }
    }

    public class TextBoxEditorView<T> : TextBoxEditorView, IViewFor<TextBoxEditorViewModel<T>>
    {
        public static readonly DependencyProperty ViewModelProperty = DependencyProperty.Register(nameof(ViewModel),
            typeof(TextBoxEditorViewModel<T>), typeof(TextBoxEditorView<TextBoxEditorViewModel<T>>));

        public TextBoxEditorViewModel<T> ViewModel
        {
            get => (TextBoxEditorViewModel<T>)GetValue(ViewModelProperty);
            set => SetValue(ViewModelProperty, value);
        }

        object IViewFor.ViewModel
        {
            get => ViewModel;
            set => ViewModel = (TextBoxEditorViewModel<T>)value;
        }

        public TextBoxEditorView()
        {
            this.WhenAnyValue(x => x.ViewModel).BindTo(this, x => x.DataContext);
        }
    }
}
