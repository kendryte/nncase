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
        public OutputConnectorViewModel From { get; }

        private InputConnectorViewModel _to;
        public InputConnectorViewModel To
        {
            get => _to;
            set
            {
                if (_to != value)
                {
                    if (_to != null)
                    {
                        _to.Connection = null;
                        _to.PositionChanged -= To_PositionChanged;
                    }

                    _to = value;
                    if (_to != null)
                    {
                        _to.Connection = this;
                        _to.PositionChanged += To_PositionChanged;
                    }

                    this.RaisePropertyChanged();
                }
            }
        }

        private Point _fromPosition;
        public Point FromPosition
        {
            get => _fromPosition;
            set
            {
                this.RaiseAndSetIfChanged(ref _fromPosition, value);
                this.RaisePropertyChanged(nameof(LabelMargin));
            }
        }

        private Point _toPosition;
        public Point ToPosition
        {
            get => _toPosition;
            set
            {
                this.RaiseAndSetIfChanged(ref _toPosition, value);
                this.RaisePropertyChanged(nameof(LabelMargin));
            }
        }

        public Thickness LabelMargin
        {
            get
            {
                var mid = ((Vector)FromPosition + (Vector)ToPosition) / 2;
                return new Thickness(mid.X + 20, mid.Y - 8, 0, 0);
            }
        }

        public ConnectionViewModel(OutputConnectorViewModel from, InputConnectorViewModel to = null)
        {
            From = from;
            From.PositionChanged += (s, e) => FromPosition = ((OutputConnectorViewModel)s).Position;
            To = to;
        }

        private void To_PositionChanged(object sender, EventArgs e)
        {
            ToPosition = ((InputConnectorViewModel)sender).Position;
        }
    }
}
