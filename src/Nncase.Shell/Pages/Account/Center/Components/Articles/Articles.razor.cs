using System.Collections.Generic;
using Nncase.Shell.Models;
using Microsoft.AspNetCore.Components;

namespace Nncase.Shell.Pages.Account.Center
{
    public partial class Articles
    {
        [Parameter] public IList<ListItemDataType> List { get; set; }
    }
}