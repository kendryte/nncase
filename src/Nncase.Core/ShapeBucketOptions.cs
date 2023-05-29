using Nncase.IR;

namespace Nncase
{
    public record ShapeBucketOptions()
    {
        public bool Enable;
        public Dictionary<Var, Expr[]> VarMap = new();
        public Dictionary<string, (int, int)> RangeInfo = new();
        public int SegmentsCount;
        public Dictionary<string, int> FixVarMap = new();
        public static ShapeBucketOptions Default => new();
    }
}
