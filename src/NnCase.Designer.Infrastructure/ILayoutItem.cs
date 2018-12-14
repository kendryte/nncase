using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ReactiveUI;

namespace NnCase.Designer
{
    public interface ILayoutItem : IReactiveObject
    {
        Guid Id { get; }

        string ContentId { get; }

        string Title { get; }

        bool IsSelected { get; set; }
    }
}
