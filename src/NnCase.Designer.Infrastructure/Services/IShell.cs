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
        event EventHandler ActiveDocumentChanged;

        ILayoutItem ActiveLayoutItem { get; }

        ObservableCollection<IDocument> Documents { get; }

        IDocument ActiveDocument { get; }

        void OpenDocument(IDocument document);

        void OpenTool<TTool>() where TTool : ITool;
    }
}
