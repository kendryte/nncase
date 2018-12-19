using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace NnCase.Designer.Modules.GraphEditor.Controls
{
    internal class ConnectorItemDragStartedEventArgs : RoutedEventArgs
    {
        public bool Cancel { get; set; }

        internal ConnectorItemDragStartedEventArgs(RoutedEvent routedEvent, object source)
            : base(routedEvent, source)
        {
        }
    }

    internal delegate void ConnectorItemDragStartedEventHandler(object sender, ConnectorItemDragStartedEventArgs e);
}
