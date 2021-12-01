using Nncase.IR;
using Nncase.IR.Math;
using F = Nncase.IR.F;
using Onnx;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitBinary(NodeProto op, BinaryOp binaryOp)
        {
            var (lhs, rhs) = GetInputExprs(op, 0, 1);
            return F.Math.Binary(binaryOp, lhs, rhs);
        }
    }
}