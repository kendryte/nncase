using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Studio.Services;

public interface IFolderPicker
{
    Task<string?> PickFolderAsync(string title, string? defaultPath = null);
}
