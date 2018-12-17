using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Services
{
    public interface IShell
    {
        ObservableCollection<IDocument> Documents { get; }

        void OpenDocument(IDocument document);
    }
}
