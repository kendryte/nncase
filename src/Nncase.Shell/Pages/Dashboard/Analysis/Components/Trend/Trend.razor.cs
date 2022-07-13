using Microsoft.AspNetCore.Components;

namespace Nncase.Shell.Pages.Dashboard.Analysis
{
    public partial class Trend
    {
        [Parameter]
        public RenderFragment ChildContent { get; set; }

        [Parameter]
        public string Flag { get; set; }
    }
}