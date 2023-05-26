using Nncase.IR;

namespace Nncase
{
    public record ShapeBucketOptions()
    {
        public bool Enable;
        public Dictionary<Var, Expr[]> VarMap;
        public Dictionary<string, (int, int)> RangeInfo;
        public int SegmentsCount;
        public Dictionary<string, int> FixVarMap;
        public static ShapeBucketOptions Default => new();
    }
}
