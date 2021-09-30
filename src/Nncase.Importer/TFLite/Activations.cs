using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private static Expr Activate(Expr input, tflite.ActivationFunctionType activation)
        {
            return activation switch
            {
                tflite.ActivationFunctionType.NONE => input,
                _ => F.Math.Clamp(input, ToFloatValueRange(activation)),
            };
        }

        private static ValueRange<float> ToFloatValueRange(tflite.ActivationFunctionType activation)
        {
            return default;
        }

        private void VisitLogistic(in tflite.Operator op)
        {
        }
    }
}
