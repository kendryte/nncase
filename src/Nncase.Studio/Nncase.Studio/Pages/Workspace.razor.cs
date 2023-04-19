using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Components;

namespace Nncase.Studio.Pages;

public partial class Workspace
{
    [Parameter]
    public string WorkingDirectory { get; set; } = default!;
}
