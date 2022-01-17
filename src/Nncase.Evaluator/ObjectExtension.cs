using System;

namespace Nncase.Evaluator
{
    public static class ObjectExtension
    {
        public static void Init(this Object obj, Action f)
        {
            if (obj == null)
            {
                f();
            }
        }
    }
}