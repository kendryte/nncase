using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace NnCase.Designer.Modules.GraphEditor.Controls
{
    internal class ConnectorItemDraggingEventArgs : RoutedEventArgs
    {
        private readonly double _horizontalChange;

        private readonly double _verticalChange;

        public ConnectorItemDraggingEventArgs(RoutedEvent routedEvent, object source, double horizontalChange, double verticalChange) :
            base(routedEvent, source)
        {
            _horizontalChange = horizontalChange;
            _verticalChange = verticalChange;
        }

        public double HorizontalChange
        {
            get { return _horizontalChange; }
        }

        public double VerticalChange
        {
            get { return _verticalChange; }
        }
    }

    internal delegate void ConnectorItemDraggingEventHandler(object sender, ConnectorItemDraggingEventArgs e);
}
