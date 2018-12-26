using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Designer.Modules.Inspector;
using NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers;
using Splat;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels
{
    public class GraphViewModel : Document
    {
        private readonly IInspectorTool _inspectorTool;

        public ObservableCollection<LayerViewModel> Layers { get; } = new ObservableCollection<LayerViewModel>();

        public ObservableCollection<ConnectionViewModel> Connections { get; } = new ObservableCollection<ConnectionViewModel>();

        public IEnumerable<LayerViewModel> SelectedLayers => Layers.Where(x => x.IsSelected);

        public GraphViewModel(string title)
        {
            Title = title;
            _inspectorTool = Locator.Current.GetService<IInspectorTool>();
        }

        public void Build(BuildGraphContext context)
        {
            var outputLayers = Layers.OfType<OutputLayerViewModel>().ToList();
            if (outputLayers.Count == 0)
            {
                MessageBox.Show("You should place at least 1 output layer.", "NnCase", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            else
            {
                foreach (var layer in outputLayers)
                    layer.BuildModel(context);
                foreach (var layer in outputLayers)
                    layer.BuildConnections(context);

                context.Graph = new Graph(context.Layers.Values.OfType<InputLayer>().ToList(),
                    context.Layers.Values.OfType<OutputLayer>().ToList());
            }
        }

        internal void OnSelectionChanged()
        {
            var selectedLayers = SelectedLayers.ToList();

            if (selectedLayers.Count == 1)
            {
                _inspectorTool.SelectedObject = new InspectableObjectBuilder()
                    .WithObjectProperties(selectedLayers[0], x => true)
                    .ToInspectableObject();
            }
            else
            {
                _inspectorTool.SelectedObject = null;
            }
        }

        internal object OnConnectionDragStarted(OutputConnectorViewModel sourceConnector, Point currentDragPoint)
        {
            var connection = new ConnectionViewModel(sourceConnector)
            {
                FromPosition = currentDragPoint
            };

            Connections.Add(connection);

            return connection;
        }

        internal void OnConnectionDragging(Point currentDragPoint, ConnectionViewModel connection)
        {
            var nearbyConnector = FindNearbyInputConnector(currentDragPoint);
            connection.ToPosition = (nearbyConnector != null)
                ? nearbyConnector.Position
                : currentDragPoint;
        }

        private InputConnectorViewModel FindNearbyInputConnector(Point mousePosition)
        {
            return Layers.SelectMany(x => x.InputConnectors)
                .FirstOrDefault(x => AreClose(x.Position, mousePosition, 10));
        }

        private static bool AreClose(Point point1, Point point2, double distance)
        {
            return (point1 - point2).Length < distance;
        }

        internal void OnConnectionDragCompleted(Point currentDragPoint, ConnectionViewModel newConnection, OutputConnectorViewModel sourceConnector)
        {
            var nearbyConnector = FindNearbyInputConnector(currentDragPoint);

            if (nearbyConnector == null || sourceConnector.Owner == nearbyConnector.Owner)
            {
                Connections.Remove(newConnection);
                return;
            }

            var existingConnection = nearbyConnector.Connection;
            if (existingConnection != null)
                Connections.Remove(existingConnection);
            
            newConnection.To = nearbyConnector;
        }
    }
}
