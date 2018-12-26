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
using NnCase.Designer.Modules.GraphEditor.Controls;
using NnCase.Designer.Modules.ModelDesigner.ViewModels;
using NnCase.Designer.Modules.Toolbox;
using NnCase.Designer.Toolbox;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.Views
{
    /// <summary>
    /// GraphView.xaml 的交互逻辑
    /// </summary>
    public partial class GraphView : ReactiveUserControl<GraphViewModel>
    {
        private Point _originalContentMouseDownPoint;

        public GraphView()
        {
            InitializeComponent();

            this.WhenActivated(d =>
            {
                this.OneWayBind(ViewModel, vm => vm.Layers, v => v._graphControl.ElementsSource)
                    .DisposeWith(d);
                this.OneWayBind(ViewModel, vm => vm.Connections, v => v._graphControl.ConnectionsSource)
                    .DisposeWith(d);
            });
        }

        protected override void OnPreviewMouseDown(MouseButtonEventArgs e)
        {
            Focus();
            base.OnPreviewMouseDown(e);
        }

        protected override void OnKeyDown(KeyEventArgs e)
        {
            //if (e.Key == Key.Delete)
            //    ((GraphViewModel)DataContext).DeleteSelectedElements();
            base.OnKeyDown(e);
        }

        private void OnGraphControlRightMouseButtonDown(object sender, MouseButtonEventArgs e)
        {
            _originalContentMouseDownPoint = e.GetPosition(_graphControl);
            _graphControl.CaptureMouse();
            Mouse.OverrideCursor = Cursors.ScrollAll;
            e.Handled = true;
        }

        private void OnGraphControlRightMouseButtonUp(object sender, MouseButtonEventArgs e)
        {
            Mouse.OverrideCursor = null;
            _graphControl.ReleaseMouseCapture();
            e.Handled = true;
        }

        private void OnGraphControlMouseMove(object sender, MouseEventArgs e)
        {
            if (e.RightButton == MouseButtonState.Pressed && _graphControl.IsMouseCaptured)
            {
                Point currentContentMousePoint = e.GetPosition(_graphControl);
                Vector dragOffset = currentContentMousePoint - _originalContentMouseDownPoint;

                _zoomAndPanControl.ContentOffsetX -= dragOffset.X;
                _zoomAndPanControl.ContentOffsetY -= dragOffset.Y;

                e.Handled = true;
            }
        }

        private void OnGraphControlMouseWheel(object sender, MouseWheelEventArgs e)
        {
            const double minScale = 0.2;
            _zoomAndPanControl.ZoomAboutPoint(
                Math.Max(minScale, _zoomAndPanControl.ContentScale + e.Delta / 1000.0f),
                e.GetPosition(_graphControl));

            e.Handled = true;
        }

        private void OnGraphControlSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            ViewModel.OnSelectionChanged();
        }

        private void OnGraphControlConnectionDragStarted(object sender, ConnectionDragStartedEventArgs e)
        {
            if (e.SourceConnector.DataContext is OutputConnectorViewModel sourceConnector)
            {
                var currentDragPoint = Mouse.GetPosition(_graphControl);
                var connection = ViewModel.OnConnectionDragStarted(sourceConnector, currentDragPoint);
                e.Connection = connection;
            }
        }

        private void OnGraphControlConnectionDragging(object sender, ConnectionDraggingEventArgs e)
        {
            var currentDragPoint = Mouse.GetPosition(_graphControl);
            var connection = (ConnectionViewModel)e.Connection;
            ViewModel.OnConnectionDragging(currentDragPoint, connection);
        }

        private void OnGraphControlConnectionDragCompleted(object sender, ConnectionDragCompletedEventArgs e)
        {
            var currentDragPoint = Mouse.GetPosition(_graphControl);
            var sourceConnector = (OutputConnectorViewModel)e.SourceConnector.DataContext;
            var newConnection = (ConnectionViewModel)e.Connection;
            ViewModel.OnConnectionDragCompleted(currentDragPoint, newConnection, sourceConnector);
        }

        private void OnGraphControlDragEnter(object sender, DragEventArgs e)
        {
            if (!e.Data.GetDataPresent(ToolboxDragDrop.DataFormat))
                e.Effects = DragDropEffects.None;
        }

        private void OnGraphControlDrop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(ToolboxDragDrop.DataFormat))
            {
                var mousePosition = e.GetPosition(_graphControl);

                var toolboxItem = (ToolboxItem)e.Data.GetData(ToolboxDragDrop.DataFormat);
                var layer = (LayerViewModel)Activator.CreateInstance(toolboxItem.ItemType);
                var namePrefix = layer.GetType().Name.Replace("ViewModel", string.Empty);
                layer.Name = GetDefaultLayerName(namePrefix);
                layer.X = mousePosition.X;
                layer.Y = mousePosition.Y;

                ViewModel.Layers.Add(layer);
            }
        }

        private string GetDefaultLayerName(string namePrefix)
        {
            int id = 0;
            string name;
            while (true)
            {
                name = namePrefix + id;
                if (ViewModel.Layers.All(x => x.Name != name)) break;
                id++;
            }

            return name;
        }
    }
}
