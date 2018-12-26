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
        public string Name { get; set; }

        public LayerViewModel Owner { get; }

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
                    var old = _connection;
                    _connection = value;
                    if (old != null)
                    {
                        old.From.Updated -= From_Updated;
                        old.To = null;
                    }

                    if (value != null)
                    {
                        value.To = this;
                        value.From.Updated += From_Updated;
                    }

                    this.RaisePropertyChanged();
                    Updated?.Invoke(this, EventArgs.Empty);
                }
            }
        }

        public event EventHandler Updated;

        public InputConnectorViewModel(string name, LayerViewModel owner)
        {
            Name = name;
            Owner = owner;
        }

        private void From_Updated(object sender, EventArgs e)
        {
            Updated?.Invoke(this, EventArgs.Empty);
        }

        public void Build(BuildGraphContext context)
        {
            var from = context.OutputConnectors[Connection.From];
            var to = context.InputConnectors[this];
            to.SetConnection(from);
        }
    }
}
