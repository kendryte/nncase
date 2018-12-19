using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels
{
    public class ConnectionViewModel : ReactiveObject
    {
        public Connection Model { get; }

        public OutputConnectorViewModel From { get; }

        public InputConnectorViewModel To { get; }

        public ConnectionViewModel(OutputConnectorViewModel from, InputConnectorViewModel to = null)
        {
            if (from != null && to != null)
                Model = new Connection(from.Model, to.Model);
            From = from;
            To = to;
        }
    }
}
