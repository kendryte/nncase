using Nncase.IR;
using Nncase.IR.Math;
using F = Nncase.IR.F;
using Onnx;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitUnary(NodeProto op, UnaryOp unaryOp)
        {
            var input = GetInputExpr(op, 0);
            return F.Math.Unary(unaryOp, input);
        }
    }
}