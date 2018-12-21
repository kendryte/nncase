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
using NnCase.Designer.Modules.Toolbox.ViewModels;
using ReactiveUI;

namespace NnCase.Designer.Modules.Toolbox.Views
{
    /// <summary>
    /// ToolboxView.xaml 的交互逻辑
    /// </summary>
    public partial class ToolboxView : ReactiveUserControl<ToolboxViewModel>
    {
        private bool _draggingItem;
        private Point _mouseStartPosition;

        public ToolboxView()
        {
            InitializeComponent();

            this.WhenActivated(d =>
            {
                this.OneWayBind(ViewModel, vm => vm.Items, v => v._toolbox.ItemsSource);
            });
        }

        private void Toolbox_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            var listBoxItem = VisualTreeUtility.FindParent<ListBoxItem>(
                (DependencyObject)e.OriginalSource);
            _draggingItem = listBoxItem != null;

            _mouseStartPosition = e.GetPosition(_toolbox);
        }

        private void Toolbox_MouseMove(object sender, MouseEventArgs e)
        {
            if (!_draggingItem)
                return;

            // Get the current mouse position
            Point mousePosition = e.GetPosition(null);
            Vector diff = _mouseStartPosition - mousePosition;

            if (e.LeftButton == MouseButtonState.Pressed &&
                (Math.Abs(diff.X) > SystemParameters.MinimumHorizontalDragDistance ||
                Math.Abs(diff.Y) > SystemParameters.MinimumVerticalDragDistance))
            {
                var listBoxItem = VisualTreeUtility.FindParent<ListBoxItem>(
                    (DependencyObject)e.OriginalSource);

                if (listBoxItem == null)
                    return;

                var itemViewModel = (ToolboxItemViewModel)_toolbox.ItemContainerGenerator.
                    ItemFromContainer(listBoxItem);

                var dragData = new DataObject(ToolboxDragDrop.DataFormat, itemViewModel.Model);
                DragDrop.DoDragDrop(listBoxItem, dragData, DragDropEffects.Move);
            }
        }
    }
}
