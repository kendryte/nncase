using CommunityToolkit.Mvvm.ComponentModel;

namespace Nncase.Studio.ViewModels;

public partial class ShapeBucketViewModel : ViewModelBase
{
    // todo: valid number
    [ObservableProperty]
    private string _segmentCount;

    [ObservableProperty]
    private string _fixVarMap;

    [ObservableProperty]
    private string _varRangeInfo;

    public ShapeBucketViewModel(ViewModelContext context)
    {
        Context = context;
    }

    public override void UpdateContext()
    {
        Context.CompileOption.ShapeBucketOptions.Enable = true;
        if (int.TryParse(SegmentCount, out var count))
        {
            Context.CompileOption.ShapeBucketOptions.SegmentsCount = count;
            // todo: other param
        }
    }
}
