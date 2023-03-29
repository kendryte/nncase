namespace Nncase
{
    public record ShapeBucketOptions(bool Enable, SegmentInfo[] SegmentInfos)
    {
        public static ShapeBucketOptions Default => new(false, Array.Empty<SegmentInfo>());
    }
}
