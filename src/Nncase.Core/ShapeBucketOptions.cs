using Nncase.IR;

namespace Nncase
{
    public record ShapeBucketOptions(bool Enable, Dictionary<Var, Expr[]> VarMap, Dictionary<string, (int, int)> RangeInfo, int SegmentsCount, Dictionary<string, int> FixVarMap)
    {
        public static ShapeBucketOptions Default => new(false, new(), new(), 0, new());
    }
}
