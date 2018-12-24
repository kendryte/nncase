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
    public class ConnectionViewModel : ReactiveObject
    {
        public Connection Model { get; }

        public OutputConnectorViewModel From { get; }

        public InputConnectorViewModel To { get; }

        private Point _fromPosition;
        public Point FromPosition
        {
            get => _fromPosition;
            set => this.RaiseAndSetIfChanged(ref _fromPosition, value);
        }

        private Point _toPosition;
        public Point ToPosition
        {
            get => _toPosition;
            set => this.RaiseAndSetIfChanged(ref _toPosition, value);
        }

        public ConnectionViewModel(OutputConnectorViewModel from, InputConnectorViewModel to = null)
        {
            if (from != null && to != null)
                Model = new Connection(from.Model, to.Model);
            From = from;
            From.PositionChanged += (s, e) => FromPosition = ((OutputConnectorViewModel)s).Position;
            To = to;
        }
    }
}
