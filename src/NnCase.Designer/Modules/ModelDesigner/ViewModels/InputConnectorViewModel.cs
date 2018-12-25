using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using NnCase.Converter.Model;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels
{
    public class InputConnectorViewModel : ReactiveObject
    {
        public InputConnector Model { get; private set; }

        public string Name => Model.Name;

        public ILayerViewModel Owner { get; }

        private Point _position;
        public Point Position
        {
            get => _position;
            set
            {
                if (_position != value)
                {
                    _position = value;
                    this.RaisePropertyChanged();
                    PositionChanged?.Invoke(this, EventArgs.Empty);
                }
            }
        }

        public event EventHandler PositionChanged;

        public bool IsInput => true;

        private ConnectionViewModel _connection;
        public ConnectionViewModel Connection
        {
            get => _connection;
            set
            {
                if (_connection != value)
                {
                    if (_connection != null)
                        _connection.To = null;
                    _connection = value;
                    if (_connection != null)
                        _connection.To = this;
                    this.RaisePropertyChanged();
                }
            }
        }

        public InputConnectorViewModel(string name, ILayerViewModel owner)
        {
            Owner = owner;
        }
    }
}
