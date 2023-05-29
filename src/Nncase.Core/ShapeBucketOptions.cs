using Nncase.IR;

namespace Nncase
{
    public record ShapeBucketOptions(bool Enable, Dictionary<Var, Expr[]> VarMap, Dictionary<string, (int Min, int Max)> RangeInfo, int SegmentsCount, Dictionary<string, int> FixVarMap)
    {
        public bool Enable = Enable;
        public Dictionary<Var, Expr[]> VarMap = VarMap;
        public Dictionary<string, (int, int)> RangeInfo = RangeInfo;
        public int SegmentsCount = SegmentsCount;
        public Dictionary<string, int> FixVarMap = FixVarMap;
        public static ShapeBucketOptions Default => new(false, new(), new(), 0, new());
    }
}
