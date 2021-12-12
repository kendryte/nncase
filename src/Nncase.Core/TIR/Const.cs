using Nncase.IR;
using static Nncase.IR.Utility;

namespace Nncase.TIR
{
    public static class ConstExtension
    {
        public static bool IsIntImm(this Const con) => (IsIntegral(DataType.Int32) & IsScalar()).MatchLeaf(con.ValueType);
        
        public static bool IsFloatImm(this Const con) => (IsFloat(DataType.Float32) & IsScalar()).MatchLeaf(con.ValueType);
    }
}