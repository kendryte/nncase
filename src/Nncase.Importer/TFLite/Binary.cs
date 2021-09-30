using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private object VisitBinary(in tflite.Operator op, BinaryOp binaryOp, tflite.ActivationFunctionType activation)
        {
            (var lhs, var rhs) = GetInputExprs(op, 0, 1);
            var node = F.Math.Binary(binaryOp, lhs, rhs);
            return Activate(node, activation);
        }
    }
}
