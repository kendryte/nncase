
using System;

namespace C
{
    /// <summary>
    /// Implements several <a href="http://en.cppreference.com/w/c/numeric/math">C Standard</a> mathematical functions that are missing from the .NET framework.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Both double and single precision functions are implemented.
    /// All functions are static and their names follow the <a href="http://en.cppreference.com/w/c/numeric/math">C Standard</a>.
    /// The class is named after the C header file where the functions are declared.
    /// </para>
    /// </remarks>
    /// <author email="robert.baron@videotron.ca">Robert Baron</author>
    public sealed class math
    {

        /// <summary>
        /// Constructor is declared <c>private</c> because all members are <c>static</c>.
        /// </summary>
        /// <remarks></remarks>
        private math()
        {
        }

        #region "The values returned by 'ilogb' for 0, NaN, and infinity respectively."

        /// <summary>
        /// Value returned by <see cref="math.ilogb(double)"/> or <see cref="ilogb(float)"/> when its input argument is <c>±0</c>.
        /// </summary>
        public const int FP_ILOGB0 = (-2147483647);

        /// <summary>
        /// Value returned by <see cref="math.ilogb(double)"/> or <see cref="ilogb(float)"/> when its input argument is <see cref="System.Double.NaN"/> or <see cref="System.Single.NaN"/> respectively.
        /// </summary>
        public const int FP_ILOGBNAN = (2147483647);

        /// <summary>
        /// Value returned by <see cref="math.ilogb(double)"/> or <see cref="ilogb(float)"/> when its input argument is <c>±infinity</c>.
        /// </summary>
        public const int INT_MAX = (32767);

        #endregion

        #region "Floating-point number categories."

        /// <summary>
        /// Indicates that a floating-point value is normal, i.e. not infinity, subnormal, NaN (not-a-number) or zero. 
        /// </summary>
        /// <seealso cref="math.fpclassify(double)"/>
        /// <seealso cref="math.fpclassify(float)"/>
        public const int FP_NORMAL = 0;

        /// <summary>
        /// Indicates that a floating-point value is subnormal. 
        /// </summary>
        /// <seealso cref="math.fpclassify(double)"/>
        /// <seealso cref="math.fpclassify(float)"/>
        public const int FP_SUBNORMAL = 1;

        /// <summary>
        /// Indicates that a floating-point value is positive or negative zero. 
        /// </summary>
        /// <seealso cref="math.fpclassify(double)"/>
        /// <seealso cref="math.fpclassify(float)"/>
        public const int FP_ZERO = 2;

        /// <summary>
        /// Indicates that the value is not representable by the underlying type (positive or negative infinity)  
        /// </summary>
        /// <seealso cref="math.fpclassify(double)"/>
        /// <seealso cref="math.fpclassify(float)"/>
        public const int FP_INFINITE = 3;

        /// <summary>
        /// Indicates that the value is not representable by the underlying type (positive or negative infinity)  
        /// </summary>
        /// <seealso cref="math.fpclassify(double)"/>
        /// <seealso cref="math.fpclassify(float)"/>
        public const int FP_NAN = 4;

        #endregion

        #region "Properties of floating-point types."

        /// <summary>
        /// The exponent bias of a <see cref="Double"/>, i.e. value to subtract from the stored exponent to get the real exponent (<c>1023</c>).
        /// </summary>
        public const int DBL_EXP_BIAS = 1023;

        /// <summary>
        /// The number of bits in the exponent of a <see cref="Double"/> (<c>11</c>).
        /// </summary>
        public const int DBL_EXP_BITS = 11;

        /// <summary>
        /// The maximum (unbiased) exponent of a <see cref="Double"/> (<c>1023</c>).
        /// </summary>
        public const int DBL_EXP_MAX = 1023;

        /// <summary>
        /// The minimum (unbiased) exponent of a <see cref="Double"/> (<c>-1022</c>).
        /// </summary>
        public const int DBL_EXP_MIN = -1022;

        /// <summary>
        /// Bit-mask used for clearing the exponent bits of a <see cref="Double"/> (<c>0x800fffffffffffff</c>).
        /// </summary>
        public const long DBL_EXP_CLR_MASK = DBL_SGN_MASK | DBL_MANT_MASK;

        /// <summary>
        /// Bit-mask used for extracting the exponent bits of a <see cref="Double"/> (<c>0x7ff0000000000000</c>).
        /// </summary>
        public const long DBL_EXP_MASK = 0x7ff0000000000000L;

        /// <summary>
        /// The number of bits in the mantissa of a <see cref="Double"/>, excludes the implicit leading <c>1</c> bit (<c>52</c>).
        /// </summary>
        public const int DBL_MANT_BITS = 52;

        /// <summary>
        /// Bit-mask used for clearing the mantissa bits of a <see cref="Double"/> (<c>0xfff0000000000000</c>).
        /// </summary>
        public const long DBL_MANT_CLR_MASK = DBL_SGN_MASK | DBL_EXP_MASK;

        /// <summary>
        /// Bit-mask used for extracting the mantissa bits of a <see cref="Double"/> (<c>0x000fffffffffffff</c>).
        /// </summary>
        public const long DBL_MANT_MASK = 0x000fffffffffffffL;

        /// <summary>
        /// Maximum positive, normal value of a <see cref="Double"/> (<c>1.7976931348623157E+308</c>).
        /// </summary>
        public const double DBL_MAX = System.Double.MaxValue;

        /// <summary>
        /// Minimum positive, normal value of a <see cref="Double"/> (<c>2.2250738585072014e-308</c>).
        /// </summary>
        public const double DBL_MIN = 2.2250738585072014e-308D;

        /// <summary>
        /// Maximum positive, subnormal value of a <see cref="Double"/> (<c>2.2250738585072009e-308</c>).
        /// </summary>
        public const double DBL_DENORM_MAX = DBL_MIN - DBL_DENORM_MIN;

        /// <summary>
        /// Minimum positive, subnormal value of a <see cref="Double"/> (<c>4.94065645841247E-324</c>).
        /// </summary>
        public const double DBL_DENORM_MIN = System.Double.Epsilon;

        /// <summary>
        /// Bit-mask used for clearing the sign bit of a <see cref="Double"/> (<c>0x7fffffffffffffff</c>).
        /// </summary>
        public const long DBL_SGN_CLR_MASK = 0x7fffffffffffffffL;

        /// <summary>
        /// Bit-mask used for extracting the sign bit of a <see cref="Double"/> (<c>0x8000000000000000</c>).
        /// </summary>
        public const long DBL_SGN_MASK = -1 - 0x7fffffffffffffffL;

        /// <summary>
        /// The exponent bias of a <see cref="Single"/>, i.e. value to subtract from the stored exponent to get the real exponent (<c>127</c>).
        /// </summary>
        public const int FLT_EXP_BIAS = 127;

        /// <summary>
        /// The number of bits in the exponent of a <see cref="Single"/> (<c>8</c>).
        /// </summary>
        public const int FLT_EXP_BITS = 8;

        /// <summary>
        /// The maximum (unbiased) exponent of a <see cref="Single"/> (<c>127</c>).
        /// </summary>
        public const int FLT_EXP_MAX = 127;

        /// <summary>
        /// The minimum (unbiased) exponent of a <see cref="Single"/> (<c>-126</c>).
        /// </summary>
        public const int FLT_EXP_MIN = -126;

        /// <summary>
        /// Bit-mask used for clearing the exponent bits of a <see cref="Single"/> (<c>0x807fffff</c>).
        /// </summary>
        public const int FLT_EXP_CLR_MASK = FLT_SGN_MASK | FLT_MANT_MASK;

        /// <summary>
        /// Bit-mask used for extracting the exponent bits of a <see cref="Single"/> (<c>0x7f800000</c>).
        /// </summary>
        public const int FLT_EXP_MASK = 0x7f800000;

        /// <summary>
        /// The number of bits in the mantissa of a <see cref="Single"/>, excludes the implicit leading <c>1</c> bit (<c>23</c>).
        /// </summary>
        public const int FLT_MANT_BITS = 23;

        /// <summary>
        /// Bit-mask used for clearing the mantissa bits of a <see cref="Single"/> (<c>0xff800000</c>).
        /// </summary>
        public const int FLT_MANT_CLR_MASK = FLT_SGN_MASK | FLT_EXP_MASK;

        /// <summary>
        /// Bit-mask used for extracting the mantissa bits of a <see cref="Single"/> (<c>0x007fffff</c>).
        /// </summary>
        public const int FLT_MANT_MASK = 0x007fffff;

        /// <summary>
        /// Maximum positive, normal value of a <see cref="Single"/> (<c>3.40282347e+38</c>).
        /// </summary>
        public const float FLT_MAX = System.Single.MaxValue;

        /// <summary>
        /// Minimum positive, normal value of a <see cref="Single"/> (<c>1.17549435e-38</c>).
        /// </summary>
        public const float FLT_MIN = 1.17549435e-38F;

        /// <summary>
        /// Maximum positive, subnormal value of a <see cref="Single"/> (<c>1.17549421e-38</c>).
        /// </summary>
        public const float FLT_DENORM_MAX = FLT_MIN - FLT_DENORM_MIN;

        /// <summary>
        /// Minimum positive, subnormal value of a <see cref="Single"/> (<c>1.401298E-45</c>).
        /// </summary>
        public const float FLT_DENORM_MIN = System.Single.Epsilon;

        /// <summary>
        /// Bit-mask used for clearing the sign bit of a <see cref="Single"/> (<c>0x7fffffff</c>).
        /// </summary>
        public const int FLT_SGN_CLR_MASK = 0x7fffffff;

        /// <summary>
        /// Bit-mask used for extracting the sign bit of a <see cref="Single"/> (<c>0x80000000</c>).
        /// </summary>
        public const int FLT_SGN_MASK = -1 - 0x7fffffff;

        #endregion

        #region "Classification."

        #region "fpclassify"

        /// <summary>
        /// Categorizes the given floating point <paramref name="number"/> into the categories: zero, subnormal, normal, infinite or NAN.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>One of <see cref="math.FP_INFINITE"/>, <see cref="math.FP_NAN"/>, <see cref="math.FP_NORMAL"/>, <see cref="math.FP_SUBNORMAL"/> or <see cref="math.FP_ZERO"/>, specifying the category of <paramref name="number"/>.</returns>
        /// <remarks>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/fpclassify">fpclassify</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static int fpclassify(double number)
        {
            long bits = System.BitConverter.DoubleToInt64Bits(number) & math.DBL_SGN_CLR_MASK;
            if (bits >= 0x7ff0000000000000L)
                return (bits & math.DBL_MANT_MASK) == 0 ? math.FP_INFINITE : math.FP_NAN;
            else if (bits < 0x0010000000000000L)
                return bits == 0 ? math.FP_ZERO : math.FP_SUBNORMAL;
            return math.FP_NORMAL;
        }

        /// <summary>
        /// Categorizes the given floating point <paramref name="number"/> into the categories: zero, subnormal, normal, infinite or NAN.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>One of <see cref="math.FP_INFINITE"/>, <see cref="math.FP_NAN"/>, <see cref="math.FP_NORMAL"/>, <see cref="math.FP_SUBNORMAL"/> or <see cref="math.FP_ZERO"/>, specifying the category of <paramref name="number"/>.</returns>
        /// <remarks>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/fpclassify">fpclassify</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static int fpclassify(float number)
        {
            int bits = SingleToInt32Bits(number) & math.FLT_SGN_CLR_MASK;
            if (bits >= 0x7f800000)
                return (bits & math.FLT_MANT_MASK) == 0 ? math.FP_INFINITE : math.FP_NAN;
            else if (bits < 0x00800000)
                return bits == 0 ? math.FP_ZERO : math.FP_SUBNORMAL;
            return math.FP_NORMAL;
        }

        #endregion

        #region "isfinite"

        /// <summary>
        /// Checks if the given <paramref name="number"/> has finite value.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns><c>true</c> if <paramref name="number"/> is finite, <c>false</c> otherwise.</returns>
        /// <remarks>
        /// <para>
        /// A floating-point number is finite if it zero, normal, or subnormal, but not infinite or NaN.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/isfinite">isfinite</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static bool isfinite(double number)
        {
            // Check the exponent part. If it is all 1's then we have infinity/NaN, i.e., not a finite value.
            return (System.BitConverter.DoubleToInt64Bits(number) & math.DBL_EXP_MASK) != math.DBL_EXP_MASK;
        }

        /// <summary>
        /// Checks if the given <paramref name="number"/> has finite value.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns><c>true</c> if <paramref name="number"/> is finite, <c>false</c> otherwise.</returns>
        /// <remarks>
        /// <para>
        /// A floating-point number is finite if it zero, normal, or subnormal, but not infinite or NaN.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/isfinite">isfinite</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static bool isfinite(float number)
        {
            // Check the exponent part. If it is all 1's then we have infinity/NaN, i.e., not a finite value.
            return (SingleToInt32Bits(number) & math.FLT_EXP_MASK) != math.FLT_EXP_MASK;
        }

        #endregion

        #region "isinf"

        /// <summary>
        /// Checks if the given <paramref name="number"/> is positive or negative infinity.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns><c>true</c> if <paramref name="number"/> has an infinite value, <c>false</c> otherwise.</returns>
        /// <remarks>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/isinf">isinf</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static bool isinf(double number)
        {
            return (System.BitConverter.DoubleToInt64Bits(number) & math.DBL_SGN_CLR_MASK) == 0x7ff0000000000000L;
        }

        /// <summary>
        /// Checks if the given <paramref name="number"/> is positive or negative infinity.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns><c>true</c> if <paramref name="number"/> has an infinite value, <c>false</c> otherwise.</returns>
        /// <remarks>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/isinf">isinf</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static bool isinf(float number)
        {
            return (SingleToInt32Bits(number) & math.FLT_SGN_CLR_MASK) == 0x7f800000;
        }

        #endregion

        #region "isnan"

        /// <summary>
        /// Checks if the given <paramref name="number"/> is NaN (Not A Number).
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns><c>true</c> if <paramref name="number"/> is NaN, <c>false</c> otherwise.</returns>
        /// <remarks>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/isnan">isnan</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static bool isnan(double number)
        {
            return (System.BitConverter.DoubleToInt64Bits(number) & math.DBL_SGN_CLR_MASK) > 0x7ff0000000000000L;
        }

        /// <summary>
        /// Checks if the given <paramref name="number"/> is NaN (Not A Number).
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns><c>true</c> if <paramref name="number"/> is NaN, <c>false</c> otherwise.</returns>
        /// <remarks>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/isnan">isnan</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static bool isnan(float number)
        {
            return (SingleToInt32Bits(number) & math.FLT_SGN_CLR_MASK) > 0x7f800000;
        }

        #endregion

        #region "isnormal"

        /// <summary>
        /// Checks if the given <paramref name="number"/> is normal.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns><c>true</c> if <paramref name="number"/> is normal, <c>false</c> otherwise.</returns>
        /// <remarks>
        /// <para>
        /// A floating-point number is normal if it is neither zero, subnormal, infinite, nor NaN.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/isnormal">isnormal</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static bool isnormal(double number)
        {
            long bits = System.BitConverter.DoubleToInt64Bits(number) & math.DBL_SGN_CLR_MASK;
            // Not infinity or NaN and greater than zero or subnormal.
            return (bits < 0x7ff0000000000000L) && (bits >= 0x0010000000000000L);
        }

        /// <summary>
        /// Checks if the given <paramref name="number"/> is normal.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns><c>true</c> if <paramref name="number"/> is normal, <c>false</c> otherwise.</returns>
        /// <remarks>
        /// <para>
        /// A floating-point number is normal if it is neither zero, subnormal, infinite, nor NaN.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/isnormal">isnormal</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static bool isnormal(float number)
        {
            int bits = SingleToInt32Bits(number) & math.FLT_SGN_CLR_MASK;
            // Not infinity or NaN and greater than zero or subnormal.
            return (bits < 0x7f800000) && (bits >= 0x00800000);
        }

        #endregion

        #region "signbit"

        /// <summary>
        /// Gets the sign bit of the specified floating-point <paramref name="number"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The sign bit of the specified floating-point <paramref name="number"/>.</returns>
        /// <remarks>
        /// <para>
        /// The function detects the sign bit of zeroes, infinities, and NaN. Along with
        /// <see cref="math.copysign(double, double)"/>, <see cref="math.signbit(double)"/> is one
        /// of the only two portable ways to examine the sign of NaN. 
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/signbit">signbit</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static int signbit(double number)
        {
            if (System.Double.IsNaN(number))
                return ((System.BitConverter.DoubleToInt64Bits(number) & math.DBL_SGN_MASK) != 0) ? 0 : 1;
            else
                return ((System.BitConverter.DoubleToInt64Bits(number) & math.DBL_SGN_MASK) != 0) ? 1 : 0;
        }

        /// <summary>
        /// Gets the sign bit of the specified floating-point <paramref name="number"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The sign bit of the specified floating-point <paramref name="number"/>.</returns>
        /// <remarks>
        /// <para>
        /// The function detects the sign bit of zeroes, infinities, and NaN. Along with
        /// <see cref="math.copysign(float, float)"/>, <see cref="math.signbit(float)"/> is one
        /// of the only two portable ways to examine the sign of NaN. 
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/signbit">signbit</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static int signbit(float number)
        {
            if (System.Double.IsNaN(number))
                return ((SingleToInt32Bits(number) & math.FLT_SGN_MASK) != 0) ? 0 : 1;
            else
                return ((SingleToInt32Bits(number) & math.FLT_SGN_MASK) != 0) ? 1 : 0;
        }

        #endregion

        #endregion

        #region "Exponential and logarithmic functions."

        #region "frexp"

        /// <summary>
        /// Decomposes the given floating-point <paramref name="number"/> into a normalized fraction and an integral power of two.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <param name="exponent">Reference to an <see cref="int"/> value to store the exponent to.</param>
        /// <returns>A <c>fraction</c> in the range <c>[0.5, 1)</c> so that <c><paramref name="number"/> = fraction * 2^<paramref name="exponent"/></c>.</returns>
        /// <remarks>
        /// <para>
        /// Special values are treated as follows.
        /// </para>
        /// <list type="bullet" >
        /// <item>If <paramref name="number"/> is <c>±0</c>, it is returned, and <c>0</c> is returned in <paramref name="exponent"/>.</item>
        /// <item>If <paramref name="number"/> is infinite, it is returned, and an undefined value is returned in <paramref name="exponent"/>.</item>
        /// <item>If <paramref name="number"/> is NaN, it is returned, and an undefined value is returned in <paramref name="exponent"/>.</item>
        /// </list>
        /// <para>
        /// </para>
        /// <para>
        /// The function <see cref="math.frexp(double, ref int)"/>, together with its dual, <see cref="math.ldexp(double, int)"/>,
        /// can be used to manipulate the representation of a floating-point number without direct bit manipulations.
        /// </para>
        /// <para>
        /// The relation of <see cref="math.frexp(double, ref int)"/> to <see cref="math.logb(double)"/> and <see cref="math.scalbn(double, int)"/> is:
        /// </para>
        /// <para>
        /// <c><paramref name="exponent"/> = (<paramref name="number"/> == 0) ? 0 : (int)(1 + <see cref="math.logb(double)">logb</see>(<paramref name="number"/>))</c><br/>
        /// <c>fraction = <see cref="math.scalbn(double, int)">scalbn</see>(<paramref name="number"/>, -<paramref name="exponent"/>)</c>
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/frexp">frexp</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.frexp(12.8D, ref exponent) = 0.8D);
        /// Assert.IsTrue(exponent = 4);
        /// 
        /// Assert.IsTrue(math.frexp(0.25D, ref exponent) == 0.5D);
        /// Assert.IsTrue(exponent == -1);
        /// 
        /// Assert.IsTrue(math.frexp(System.Math.Pow(2D, 1023), ref exponent) == 0.5D);
        /// Assert.IsTrue(exponent == 1024);
        /// 
        /// Assert.IsTrue(math.frexp(-System.Math.Pow(2D, -1074), ref exponent) == -0.5D);
        /// Assert.IsTrue(exponent == -1073);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.frexp(12.8D, exponent) = 0.8D);
        /// Assert.IsTrue(exponent = 4);
        /// 
        /// Assert.IsTrue(math.frexp(0.25D, exponent) = 0.5D);
        /// Assert.IsTrue(exponent = -1);
        /// 
        /// Assert.IsTrue(math.frexp(System.Math.Pow(2D, 1023), exponent) = 0.5D);
        /// Assert.IsTrue(exponent = 1024);
        /// 
        /// Assert.IsTrue(math.frexp(-System.Math.Pow(2D, -1074), exponent) = -0.5D);
        /// Assert.IsTrue(exponent = -1073);
        /// </code> 
        /// </example>
        public static double frexp(double number, out int exponent)
        {
            long bits = System.BitConverter.DoubleToInt64Bits(number);
            int exp = (int)((bits & math.DBL_EXP_MASK) >> math.DBL_MANT_BITS);
            exponent = 0;

            if (exp == 0x7ff || number == 0D)
                number += number;
            else
            {
                // Not zero and finite.
                exponent = exp - 1022;
                if (exp == 0)
                {
                    // Subnormal, scale number so that it is in [1, 2).
                    number *= System.BitConverter.Int64BitsToDouble(0x4350000000000000L); // 2^54
                    bits = System.BitConverter.DoubleToInt64Bits(number);
                    exp = (int)((bits & math.DBL_EXP_MASK) >> math.DBL_MANT_BITS);
                    exponent = exp - 1022 - 54;
                }
                // Set exponent to -1 so that number is in [0.5, 1).
                number = System.BitConverter.Int64BitsToDouble((bits & math.DBL_EXP_CLR_MASK) | 0x3fe0000000000000L);
            }

            return number;
        }

        /// <summary>
        /// Decomposes the given floating-point <paramref name="number"/> into a normalized fraction and an integral power of two.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <param name="exponent">Reference to an <see cref="int"/> value to store the exponent to.</param>
        /// <returns>A <c>fraction</c> in the range <c>[0.5, 1)</c> so that <c><paramref name="number"/> = fraction * 2^<paramref name="exponent"/></c>.</returns>
        /// <remarks>
        /// <para>
        /// Special values are treated as follows.
        /// </para>
        /// <list type="bullet" >
        /// <item>If <paramref name="number"/> is <c>±0</c>, it is returned, and <c>0</c> is returned in <paramref name="exponent"/>.</item>
        /// <item>If <paramref name="number"/> is infinite, it is returned, and an undefined value is returned in <paramref name="exponent"/>.</item>
        /// <item>If <paramref name="number"/> is NaN, it is returned, and an undefined value is returned in <paramref name="exponent"/>.</item>
        /// </list>
        /// <para>
        /// The function <see cref="math.frexp(float, ref int)"/>, together with its dual, <see cref="math.ldexp(float, int)"/>,
        /// can be used to manipulate the representation of a floating-point number without direct bit manipulations.
        /// </para>
        /// <para>
        /// The relation of <see cref="math.frexp(float, ref int)"/> to <see cref="math.logb(float)"/> and <see cref="math.scalbn(float, int)"/> is:
        /// </para>
        /// <para>
        /// <c><paramref name="exponent"/> = (<paramref name="number"/> == 0) ? 0 : (int)(1 + <see cref="math.logb(float)">logb</see>(<paramref name="number"/>))</c><br/>
        /// <c>fraction = <see cref="math.scalbn(float, int)">scalbn</see>(<paramref name="number"/>, -<paramref name="exponent"/>)</c>
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/frexp">frexp</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.frexp(12.8F, ref exponent) = 0.8F);
        /// Assert.IsTrue(exponent = 4);
        /// 
        /// Assert.IsTrue(math.frexp(0.25F, ref exponent) == 0.5F);
        /// Assert.IsTrue(exponent == -1);
        /// 
        /// Assert.IsTrue(math.frexp(System.Math.Pow(2F, 127F), ref exponent) == 0.5F);
        /// Assert.IsTrue(exponent == 128);
        /// 
        /// Assert.IsTrue(math.frexp(-System.Math.Pow(2F, -149F), ref exponent) == -0.5F);
        /// Assert.IsTrue(exponent == -148);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.frexp(12.8F, exponent) = 0.8F);
        /// Assert.IsTrue(exponent = 4);
        /// 
        /// Assert.IsTrue(math.frexp(0.25F, exponent) = 0.5F);
        /// Assert.IsTrue(exponent = -1);
        /// 
        /// Assert.IsTrue(math.frexp(System.Math.Pow(2F, 127F), exponent) = 0.5F);
        /// Assert.IsTrue(exponent = 128);
        /// 
        /// Assert.IsTrue(math.frexp(-System.Math.Pow(2F, -149F), exponent) = -0.5F);
        /// Assert.IsTrue(exponent = -148);
        /// </code> 
        /// </example>
        public static float frexp(float number, out int exponent)
        {
            int bits = math.SingleToInt32Bits(number);
            int exp = (int)((bits & math.FLT_EXP_MASK) >> math.FLT_MANT_BITS);
            exponent = 0;

            if (exp == 0xff || number == 0F)
                number += number;
            else
            {
                // Not zero and finite.
                exponent = exp - 126;
                if (exp == 0)
                {
                    // Subnormal, scale number so that it is in [1, 2).
                    number *= math.Int32BitsToSingle(0x4c000000); // 2^25
                    bits = math.SingleToInt32Bits(number);
                    exp = (int)((bits & math.FLT_EXP_MASK) >> math.FLT_MANT_BITS);
                    exponent = exp - 126 - 25;
                }
                // Set exponent to -1 so that number is in [0.5, 1).
                number = math.Int32BitsToSingle((bits & math.FLT_EXP_CLR_MASK) | 0x3f000000);
            }

            return number;
        }

        #endregion

        #region "ilogb"

        /// <summary>
        /// Gets the unbiased exponent of the specified floating-point <paramref name="number"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The unbiased exponent of the specified floating-point <paramref name="number"/>, or a special value if <paramref name="number"/> is not normal or subnormal.</returns>
        /// <remarks>
        /// <para>
        /// The unbiased exponent is the integral part of the logarithm base 2 of <paramref name="number"/>.
        /// The unbiased exponent is such that:
        /// </para>
        /// <para>
        /// <c><paramref name="number"/> = <see cref="math.significand(double)">significand</see>(<paramref name="number"/>) * 2^<see cref="math.ilogb(double)">ilogb</see>(<paramref name="number"/>)</c>.
        /// </para>
        /// <para>
        /// The return unbiased exponent is valid for all normal and subnormal numbers. Special values are treated as follows.
        /// </para>
        /// <list type="bullet">
        /// <item>If <paramref name="number"/> is <c>±0</c>, <see cref="math.FP_ILOGB0"/> is returned.</item>
        /// <item>If <paramref name="number"/> is infinite, <see cref="math.INT_MAX"/> is returned.</item>
        /// <item>If <paramref name="number"/> is NaN, <see cref="math.FP_ILOGBNAN"/> is returned.</item>
        /// </list>
        /// <para>
        /// If <paramref name="number"/> is not zero, infinite, or NaN, the value returned is exactly equivalent to
        /// <c>(<see cref="int"/>)<see cref="math.logb(double)">logb</see>(<paramref name="number"/>)</c>. 
        /// </para>
        /// <para>
        /// The value of the exponent returned by <see cref="math.ilogb(double)"/> is always <c>1</c> less than the exponent retuned by
        /// <see cref="math.frexp(double, ref int)"/> because of the different normalization requirements:
        /// for <see cref="math.ilogb(double)"/>, the normalized significand is in the interval <c>[1, 2)</c>,
        /// but for <see cref="math.frexp(double, ref int)"/>, the normalized significand is in the interval <c>[0.5, 1)</c>.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/ilogb">ilogb</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.ilogb(1D) == 0);
        /// Assert.IsTrue(math.ilogb(System.Math.E) == 1);
        /// Assert.IsTrue(math.ilogb(1024D) == 10);
        /// Assert.IsTrue(math.ilogb(-2000D) == 10);
        /// 
        /// Assert.IsTrue(math.ilogb(2D) == 1);
        /// Assert.IsTrue(math.ilogb(Math.Pow(2D, 56D)) == 56);
        /// Assert.IsTrue(math.ilogb(1.1D * Math.Pow(2D, -1074D)) == -1074);
        /// Assert.IsTrue(math.ilogb(Math.Pow(2D, -1075D)) == math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb(Math.Pow(2D, 1024D)) == math.INT_MAX);
        /// Assert.IsTrue(math.ilogb(Math.Pow(2D, 1023D)) == 1023);
        /// Assert.IsTrue(math.ilogb(2D * Math.Pow(2D, 102D)) == 103);
        /// 
        /// Assert.IsTrue(math.ilogb(math.DBL_DENORM_MIN) == math.DBL_EXP_MIN - math.DBL_MANT_BITS);
        /// Assert.IsTrue(math.ilogb(math.DBL_DENORM_MAX) == math.DBL_EXP_MIN - 1);
        /// Assert.IsTrue(math.ilogb(math.DBL_MIN) == math.DBL_EXP_MIN);
        /// Assert.IsTrue(math.ilogb(math.DBL_MAX) == math.DBL_EXP_MAX);
        /// 
        /// Assert.IsTrue(math.ilogb(System.Double.PositiveInfinity) == math.INT_MAX);
        /// Assert.IsTrue(math.ilogb(System.Double.NegativeInfinity) == math.INT_MAX);
        /// Assert.IsTrue(math.ilogb(0D) == math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb(-0D) == math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb(System.Double.NaN) == math.FP_ILOGBNAN);
        /// Assert.IsTrue(math.ilogb(-System.Double.NaN) == math.FP_ILOGBNAN);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.ilogb(1D) = 0);
        /// Assert.IsTrue(math.ilogb(System.Math.E) = 1);
        /// Assert.IsTrue(math.ilogb(1024D) = 10);
        /// Assert.IsTrue(math.ilogb(-2000D) = 10);
        /// 
        /// Assert.IsTrue(math.ilogb(2D) = 1);
        /// Assert.IsTrue(math.ilogb(Math.Pow(2D, 56D)) = 56);
        /// Assert.IsTrue(math.ilogb(1.1D * Math.Pow(2D, -1074D)) = -1074);
        /// Assert.IsTrue(math.ilogb(Math.Pow(2D, -1075D)) = math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb(Math.Pow(2D, 1024D)) = math.INT_MAX);
        /// Assert.IsTrue(math.ilogb(Math.Pow(2D, 1023D)) = 1023);
        /// Assert.IsTrue(math.ilogb(2D * Math.Pow(2D, 102D)) = 103);
        /// 
        /// Assert.IsTrue(math.ilogb(math.DBL_DENORM_MIN) = math.DBL_EXP_MIN - math.DBL_MANT_BITS);
        /// Assert.IsTrue(math.ilogb(math.DBL_DENORM_MAX) = math.DBL_EXP_MIN - 1);
        /// Assert.IsTrue(math.ilogb(math.DBL_MIN) = math.DBL_EXP_MIN);
        /// Assert.IsTrue(math.ilogb(math.DBL_MAX) = math.DBL_EXP_MAX);
        /// 
        /// Assert.IsTrue(math.ilogb(System.Double.PositiveInfinity) = math.INT_MAX);
        /// Assert.IsTrue(math.ilogb(System.Double.NegativeInfinity) = math.INT_MAX);
        /// Assert.IsTrue(math.ilogb(0D) = math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb(-0D) = math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb(System.Double.NaN) = math.FP_ILOGBNAN);
        /// Assert.IsTrue(math.ilogb(-System.Double.NaN) = math.FP_ILOGBNAN);
        /// </code> 
        /// </example>
        public static int ilogb(double number)
        {
            long bits = System.BitConverter.DoubleToInt64Bits(number) & (math.DBL_EXP_MASK | math.DBL_MANT_MASK);
            if (bits == 0L)
                return math.FP_ILOGB0;
            int exp = (int)(bits >> math.DBL_MANT_BITS);
            if (exp == 0x7ff)
                return (bits & math.DBL_MANT_MASK) == 0L ? math.INT_MAX : math.FP_ILOGBNAN;
            if (exp == 0)
                exp -= (_leadingZeroesCount(bits) - (DBL_EXP_BITS + 1));
            return exp - math.DBL_EXP_BIAS;
        }

        /// <summary>
        /// Gets the unbiased exponent of the specified floating-point <paramref name="number"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The unbiased exponent of the specified floating-point <paramref name="number"/>, or a special value if <paramref name="number"/> is not normal or subnormal.</returns>
        /// <remarks>
        /// <para>
        /// The unbiased exponent is the integral part of the logarithm base 2 of <paramref name="number"/>.
        /// The unbiased exponent is such that:
        /// </para>
        /// <para>
        /// <c><paramref name="number"/> = <see cref="math.significand(float)">significand</see>(<paramref name="number"/>) * 2^<see cref="math.ilogb(float)">ilogb</see>(<paramref name="number"/>)</c>.
        /// </para>
        /// <para>
        /// The return unbiased exponent is valid for all normal and subnormal numbers. Special values are treated as follows.
        /// </para>
        /// <list type="bullet">
        /// <item>If <paramref name="number"/> is <c>±0</c>, <see cref="math.FP_ILOGB0"/> is returned.</item>
        /// <item>If <paramref name="number"/> is infinite, <see cref="math.INT_MAX"/> is returned.</item>
        /// <item>If <paramref name="number"/> is NaN, <see cref="math.FP_ILOGBNAN"/> is returned.</item>
        /// </list>
        /// <para>
        /// If <paramref name="number"/> is not zero, infinite, or NaN, the value returned is exactly equivalent to
        /// <c>(<see cref="int"/>)<see cref="math.logb(float)">logb</see>(<paramref name="number"/>)</c>. 
        /// </para>
        /// <para>
        /// The value of the exponent returned by <see cref="math.ilogb(float)"/> is always <c>1</c> less than the exponent retuned by
        /// <see cref="math.frexp(float, ref int)"/> because of the different normalization requirements:
        /// for <see cref="math.ilogb(float)"/>, the normalized significand is in the interval <c>[1, 2)</c>,
        /// but for <see cref="math.frexp(float, ref int)"/>, the normalized significand is in the interval <c>[0.5, 1)</c>.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/ilogb">ilogb</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.ilogb(1F) == 0);
        /// Assert.IsTrue(math.ilogb((float)System.Math.E) == 1);
        /// Assert.IsTrue(math.ilogb(1024F) == 10);
        /// Assert.IsTrue(math.ilogb(-2000F) == 10);
        /// 
        /// Assert.IsTrue(math.ilogb(2F) == 1);
        /// Assert.IsTrue(math.ilogb((float)Math.Pow(2F, 56F)) == 56);
        /// Assert.IsTrue(math.ilogb(1.1F * (float)Math.Pow(2F, -149F)) == -149);
        /// Assert.IsTrue(math.ilogb((float)Math.Pow(2F, -150F)) == math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb((float)Math.Pow(2F, 128F)) == math.INT_MAX);
        /// Assert.IsTrue(math.ilogb((float)Math.Pow(2D, 127F)) == 127);
        /// Assert.IsTrue(math.ilogb(2F * (float)Math.Pow(2F, 102F)) == 103);
        /// 
        /// Assert.IsTrue(math.ilogb(math.FLT_DENORM_MIN) == math.FLT_EXP_MIN - math.FLT_MANT_BITS);
        /// Assert.IsTrue(math.ilogb(math.FLT_DENORM_MAX) == math.FLT_EXP_MIN - 1);
        /// Assert.IsTrue(math.ilogb(math.FLT_MIN) == math.FLT_EXP_MIN);
        /// Assert.IsTrue(math.ilogb(math.FLT_MAX) == math.FLT_EXP_MAX);
        /// 
        /// Assert.IsTrue(math.ilogb(System.Single.PositiveInfinity) == math.INT_MAX);
        /// Assert.IsTrue(math.ilogb(System.Single.NegativeInfinity) == math.INT_MAX);
        /// Assert.IsTrue(math.ilogb(0F) == math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb(-0F) == math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb(System.Single.NaN) == math.FP_ILOGBNAN);
        /// Assert.IsTrue(math.ilogb(-System.Single.NaN) == math.FP_ILOGBNAN);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.ilogb(1F) = 0);
        /// Assert.IsTrue(math.ilogb(CSng(System.Math.E)) = 1);
        /// Assert.IsTrue(math.ilogb(1024F) = 10);
        /// Assert.IsTrue(math.ilogb(-2000F) = 10);
        /// 
        /// Assert.IsTrue(math.ilogb(2F) = 1);
        /// Assert.IsTrue(math.ilogb(CSng(Math.Pow(2F, 56F))) = 56);
        /// Assert.IsTrue(math.ilogb(1.1F * CSng(Math.Pow(2F, -149F))) = -149);
        /// Assert.IsTrue(math.ilogb(CSng(Math.Pow(2F, -150F))) = math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb(CSng(Math.Pow(2F, 128F))) = math.INT_MAX);
        /// Assert.IsTrue(math.ilogb(CSng(Math.Pow(2D, 127F))) = 127);
        /// Assert.IsTrue(math.ilogb(2F * CSng(Math.Pow(2F, 102F))) = 103);
        /// 
        /// Assert.IsTrue(math.ilogb(math.FLT_DENORM_MIN) = math.FLT_EXP_MIN - math.FLT_MANT_BITS);
        /// Assert.IsTrue(math.ilogb(math.FLT_DENORM_MAX) = math.FLT_EXP_MIN - 1);
        /// Assert.IsTrue(math.ilogb(math.FLT_MIN) = math.FLT_EXP_MIN);
        /// Assert.IsTrue(math.ilogb(math.FLT_MAX) = math.FLT_EXP_MAX);
        /// 
        /// Assert.IsTrue(math.ilogb(System.Single.PositiveInfinity) = math.INT_MAX);
        /// Assert.IsTrue(math.ilogb(System.Single.NegativeInfinity) = math.INT_MAX);
        /// Assert.IsTrue(math.ilogb(0F) = math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb(-0F) = math.FP_ILOGB0);
        /// Assert.IsTrue(math.ilogb(System.Single.NaN) = math.FP_ILOGBNAN);
        /// Assert.IsTrue(math.ilogb(-System.Single.NaN) = math.FP_ILOGBNAN);
        /// </code> 
        /// </example>
        public static int ilogb(float number)
        {
            int bits = math.SingleToInt32Bits(number) & (math.FLT_EXP_MASK | math.FLT_MANT_MASK);
            if (bits == 0L)
                return math.FP_ILOGB0;
            int exp = (bits >> math.FLT_MANT_BITS);
            if (exp == 0xff)
                return (bits & math.FLT_MANT_MASK) == 0L ? math.INT_MAX : math.FP_ILOGBNAN;
            if (exp == 0)
                exp -= (_leadingZeroesCount(bits) - (FLT_EXP_BITS + 1));
            return exp - math.FLT_EXP_BIAS;
        }

        #endregion

        #region "ldexp"

        /// <summary>
        /// Scales the specified floating-point <paramref name="number"/> by 2^<paramref name="exponent"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <param name="exponent">The exponent of the power of two.</param>
        /// <returns>The value <c><paramref name="number"/> * 2^<paramref name="exponent"/></c>.</returns>
        /// <remarks>
        /// <para>
        /// Special values are treated as follows.
        /// </para>
        /// <list type="bullet">
        /// <item>If <paramref name="number"/> is <c>±0</c>, it is returned.</item>
        /// <item>If <paramref name="number"/> is infinite, it is returned.</item>
        /// <item>If <paramref name="exponent"/> is <c>0</c>, <paramref name="number"/> is returned.</item>
        /// <item>If <paramref name="number"/> is NaN, <see cref="System.Double.NaN"/> is returned.</item>
        /// </list>
        /// <para>
        /// The function <see cref="math.ldexp(double, int)"/> ("load exponent"), together with its dual, <see cref="math.frexp(double, ref int)"/>,
        /// can be used to manipulate the representation of a floating-point number without direct bit manipulations.
        /// </para>
        /// <para>
        /// The function <see cref="math.ldexp(double, int)"/> is equivalent to <see cref="math.scalbn(double, int)"/>.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/ldexp">ldexp</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.ldexp(0.8D, 4) == 12.8D);
        /// Assert.IsTrue(math.ldexp(-0.854375D, 5) == -27.34D);
        /// Assert.IsTrue(math.ldexp(1D, 0) == 1D);
        /// 
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN / 2D, 0) == math.DBL_MIN / 2D);
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN / 2D, 1) == math.DBL_MIN);
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS) == 2D * math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS - 1) == math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS) == math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS - 1) == math.DBL_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.ldexp(1D, System.Int32.MaxValue) == System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.ldexp(1D, System.Int32.MinValue) == 0D);
        /// Assert.IsTrue(math.ldexp(-1D, System.Int32.MaxValue) == System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.ldexp(-1D, System.Int32.MinValue) == -0D);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.ldexp(0.8D, 4) = 12.8D);
        /// Assert.IsTrue(math.ldexp(-0.854375D, 5) = -27.34D);
        /// Assert.IsTrue(math.ldexp(1D, 0) = 1D);
        /// 
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN / 2D, 0) = math.DBL_MIN / 2D);
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN / 2D, 1) = math.DBL_MIN);
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS) = 2D * math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS - 1) = math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS) = math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS - 1) = math.DBL_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.ldexp(1D, System.Int32.MaxValue) = System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.ldexp(1D, System.Int32.MinValue) = 0D);
        /// Assert.IsTrue(math.ldexp(-1D, System.Int32.MaxValue) = System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.ldexp(-1D, System.Int32.MinValue) = -0D);
        /// </code> 
        /// </example>
        public static double ldexp(double number, int exponent)
        {
            return scalbn(number, exponent);
        }

        /// <summary>
        /// Scales the specified floating-point <paramref name="number"/> by 2^<paramref name="exponent"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <param name="exponent">The exponent of the power of two.</param>
        /// <returns>The value <c><paramref name="number"/> * 2^<paramref name="exponent"/></c>.</returns>
        /// <remarks>
        /// <para>
        /// Special values are treated as follows.
        /// </para>
        /// <list type="bullet">
        /// <item>If <paramref name="number"/> is <c>±0</c>, it is returned.</item>
        /// <item>If <paramref name="number"/> is infinite, it is returned.</item>
        /// <item>If <paramref name="exponent"/> is <c>0</c>, <paramref name="number"/> is returned.</item>
        /// <item>If <paramref name="number"/> is NaN, <see cref="System.Single.NaN"/> is returned.</item>
        /// </list>
        /// <para>
        /// The function <see cref="math.ldexp(float, int)"/> ("load exponent"), together with its dual, <see cref="math.frexp(float, ref int)"/>,
        /// can be used to manipulate the representation of a floating-point number without direct bit manipulations.
        /// </para>
        /// <para>
        /// The function <see cref="math.ldexp(float, int)"/> is equivalent to <see cref="math.scalbn(float, int)"/>.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/ldexp">ldexp</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.ldexp(0.8F, 4) == 12.8F);
        /// Assert.IsTrue(math.ldexp(-0.854375F, 5) == -27.34F);
        /// Assert.IsTrue(math.ldexp(1F, 0) == 1F);
        /// 
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN / 2F, 0) == math.FLT_MIN / 2F);
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN / 2F, 1) == math.FLT_MIN);
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS) == 2F * math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS - 1) == math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS) == math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS - 1) == math.FLT_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.ldexp(1F, System.Int32.MaxValue) == System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.ldexp(1F, System.Int32.MinValue) == 0F);
        /// Assert.IsTrue(math.ldexp(-1F, System.Int32.MaxValue) == System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.ldexp(-1F, System.Int32.MinValue) == -0F);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.ldexp(0.8F, 4) = 12.8F);
        /// Assert.IsTrue(math.ldexp(-0.854375F, 5) = -27.34F);
        /// Assert.IsTrue(math.ldexp(1F, 0) = 1F);
        /// 
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN / 2F, 0) = math.FLT_MIN / 2F);
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN / 2F, 1) = math.FLT_MIN);
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS) = 2F * math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS - 1) = math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS) = math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.ldexp(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS - 1) = math.FLT_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.ldexp(1F, System.Int32.MaxValue) = System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.ldexp(1F, System.Int32.MinValue) = 0F);
        /// Assert.IsTrue(math.ldexp(-1F, System.Int32.MaxValue) = System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.ldexp(-1F, System.Int32.MinValue) = -0F);
        /// </code> 
        /// </example>
        public static float ldexp(float number, int exponent)
        {
            return scalbn(number, exponent);
        }

        #endregion

        #region "logb"

        /// <summary>
        /// Gets the unbiased exponent of the specified floating-point <paramref name="number"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The unbiased exponent of the specified floating-point <paramref name="number"/>, or a special value if <paramref name="number"/> is not normal or subnormal.</returns>
        /// <remarks>
        /// <para>
        /// The unbiased exponent is the integral part of the logarithm base 2 of <paramref name="number"/>.
        /// The unbiased exponent is such that
        /// </para>
        /// <para>
        /// <c><paramref name="number"/> = <see cref="math.significand(double)">significand</see>(<paramref name="number"/>) * 2^<see cref="math.logb(double)">logb</see>(<paramref name="number"/>)</c>.
        /// </para>
        /// <para>
        /// The return unbiased exponent is valid for all normal and subnormal numbers. Special values are treated as follows.
        /// </para>
        /// <list type="bullet">
        /// <item>If <paramref name="number"/> is <c>±0</c>, <see cref="System.Double.NegativeInfinity"/> is returned.</item>
        /// <item>If <paramref name="number"/> is infinite, <see cref="System.Double.PositiveInfinity"/> is returned.</item>
        /// <item>If <paramref name="number"/> is NaN, <see cref="System.Double.NaN"/> is returned.</item>
        /// </list>
        /// <para>
        /// If <paramref name="number"/> is not zero, infinite, or NaN, the value returned is exactly equivalent to
        /// <c><see cref="math.ilogb(double)">ilogb</see>(<paramref name="number"/>)</c>. 
        /// </para>
        /// <para>
        /// The value of the exponent returned by <see cref="math.logb(double)"/> is always <c>1</c> less than the exponent retuned by
        /// <see cref="math.frexp(double, ref int)"/> because of the different normalization requirements:
        /// for <see cref="math.logb(double)"/>, the normalized significand is in the interval <c>[1, 2)</c>,
        /// but for <see cref="math.frexp(double, ref int)"/>, the normalized significand is in the interval <c>[0.5, 1)</c>. 
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/logb">logb</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.logb(1D) == 0D);
        /// Assert.IsTrue(math.logb(System.Math.E) == 1D);
        /// Assert.IsTrue(math.logb(1024D) == 10D);
        /// Assert.IsTrue(math.logb(-2000D) == 10D);
        /// 
        /// Assert.IsTrue(math.logb(2D) == 1D);
        /// Assert.IsTrue(math.logb(Math.Pow(2D, 56D)) == 56D);
        /// Assert.IsTrue(math.logb(1.1D * Math.Pow(2D, -1074D)) == -1074D);
        /// Assert.IsTrue(math.logb(Math.Pow(2D, -1075D)) == System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.logb(Math.Pow(2D, 1024D)) == System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.logb(Math.Pow(2D, 1023D)) == 1023D);
        /// Assert.IsTrue(math.logb(2D * Math.Pow(2D, 102D)) == 103D);
        /// 
        /// Assert.IsTrue(math.logb(math.DBL_DENORM_MIN) == math.DBL_EXP_MIN - math.DBL_MANT_BITS);
        /// Assert.IsTrue(math.logb(math.DBL_DENORM_MAX) == math.DBL_EXP_MIN - 1);
        /// Assert.IsTrue(math.logb(math.DBL_MIN) == math.DBL_EXP_MIN);
        /// Assert.IsTrue(math.logb(math.DBL_MAX) == math.DBL_EXP_MAX);
        /// 
        /// Assert.IsTrue(math.logb(System.Double.PositiveInfinity) == System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.logb(System.Double.NegativeInfinity) == System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.logb(0D) == System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.logb(-0D) == System.Double.NegativeInfinity);
        /// Assert.IsTrue(System.Double.IsNaN(math.logb(System.Double.NaN)));
        /// Assert.IsTrue(System.Double.IsNaN(math.logb(-System.Double.NaN)));
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.logb(1D) = 0D);
        /// Assert.IsTrue(math.logb(System.Math.E) = 1D);
        /// Assert.IsTrue(math.logb(1024D) = 10D);
        /// Assert.IsTrue(math.logb(-2000D) = 10D);
        /// 
        /// Assert.IsTrue(math.logb(2D) = 1D);
        /// Assert.IsTrue(math.logb(Math.Pow(2D, 56D)) = 56D);
        /// Assert.IsTrue(math.logb(1.1D * Math.Pow(2D, -1074D)) = -1074D);
        /// Assert.IsTrue(math.logb(Math.Pow(2D, -1075D)) = System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.logb(Math.Pow(2D, 1024D)) = System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.logb(Math.Pow(2D, 1023D)) = 1023D);
        /// Assert.IsTrue(math.logb(2D * Math.Pow(2D, 102D)) = 103D);
        /// 
        /// Assert.IsTrue(math.logb(math.DBL_DENORM_MIN) = math.DBL_EXP_MIN - math.DBL_MANT_BITS);
        /// Assert.IsTrue(math.logb(math.DBL_DENORM_MAX) = math.DBL_EXP_MIN - 1);
        /// Assert.IsTrue(math.logb(math.DBL_MIN) = math.DBL_EXP_MIN);
        /// Assert.IsTrue(math.logb(math.DBL_MAX) = math.DBL_EXP_MAX);
        /// 
        /// Assert.IsTrue(math.logb(System.Double.PositiveInfinity) = System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.logb(System.Double.NegativeInfinity) = System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.logb(0D) = System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.logb(-0D) = System.Double.NegativeInfinity);
        /// Assert.IsTrue(System.Double.IsNaN(math.logb(System.Double.NaN)));
        /// Assert.IsTrue(System.Double.IsNaN(math.logb(-System.Double.NaN)));
        /// </code> 
        /// </example>
        public static double logb(double number)
        {
            int exp = math.ilogb(number);
            switch (exp)
            {
                case math.FP_ILOGB0:
                    return System.Double.NegativeInfinity;
                case math.FP_ILOGBNAN:
                    return System.Double.NaN;
                case math.INT_MAX:
                    return System.Double.PositiveInfinity;
            }
            return exp;
        }

        /// <summary>
        /// Gets the unbiased exponent of the specified floating-point <paramref name="number"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The unbiased exponent of the specified floating-point <paramref name="number"/>, or a special value if <paramref name="number"/> is not normal or subnormal.</returns>
        /// <remarks>
        /// <para>
        /// The unbiased exponent is the integral part of the logarithm base 2 of <paramref name="number"/>.
        /// The unbiased exponent is such that
        /// </para>
        /// <para>
        /// <c><paramref name="number"/> = <see cref="math.significand(float)">significand</see>(<paramref name="number"/>) * 2^<see cref="math.logb(float)">logb</see>(<paramref name="number"/>)</c>.
        /// </para>
        /// <para>
        /// The return unbiased exponent is valid for all normal and subnormal numbers. Special values are treated as follows.
        /// </para>
        /// <list type="bullet">
        /// <item>If <paramref name="number"/> is <c>±0</c>, <see cref="System.Single.NegativeInfinity"/> is returned.</item>
        /// <item>If <paramref name="number"/> is infinite, <see cref="System.Single.PositiveInfinity"/> is returned.</item>
        /// <item>If <paramref name="number"/> is NaN, <see cref="System.Single.NaN"/> is returned.</item>
        /// </list>
        /// <para>
        /// If <paramref name="number"/> is not zero, infinite, or NaN, the value returned is exactly equivalent to
        /// <c><see cref="math.ilogb(float)">ilogb</see>(<paramref name="number"/>)</c>. 
        /// </para>
        /// <para>
        /// The value of the exponent returned by <see cref="math.logb(float)"/> is always <c>1</c> less than the exponent retuned by
        /// <see cref="math.frexp(float, ref int)"/> because of the different normalization requirements:
        /// for <see cref="math.logb(float)"/>, the normalized significand is in the interval <c>[1, 2)</c>,
        /// but for <see cref="math.frexp(float, ref int)"/>, the normalized significand is in the interval <c>[0.5, 1)</c>. 
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/logb">logb</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.logb(1F) == 0F);
        /// Assert.IsTrue(math.logb((float)System.Math.E) == 1F);
        /// Assert.IsTrue(math.logb(1024F) == 10F);
        /// Assert.IsTrue(math.logb(-2000F) == 10F);
        /// 
        /// Assert.IsTrue(math.logb(2F) == 1F);
        /// Assert.IsTrue(math.logb((float)Math.Pow(2F, 56F)) == 56F);
        /// Assert.IsTrue(math.logb(1.1F * (float)Math.Pow(2F, -149F)) == -149F);
        /// Assert.IsTrue(math.logb((float)Math.Pow(2F, -150F)) == System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.logb((float)Math.Pow(2F, 128F)) == System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.logb((float)Math.Pow(2D, 127F)) == 127F);
        /// Assert.IsTrue(math.logb(2F * (float)Math.Pow(2F, 102F)) == 103F);
        /// 
        /// Assert.IsTrue(math.logb(math.FLT_DENORM_MIN) == math.FLT_EXP_MIN - math.FLT_MANT_BITS);
        /// Assert.IsTrue(math.logb(math.FLT_DENORM_MAX) == math.FLT_EXP_MIN - 1);
        /// Assert.IsTrue(math.logb(math.FLT_MIN) == math.FLT_EXP_MIN);
        /// Assert.IsTrue(math.logb(math.FLT_MAX) == math.FLT_EXP_MAX);
        /// 
        /// Assert.IsTrue(math.logb(System.Single.PositiveInfinity) == System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.logb(System.Single.NegativeInfinity) == System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.logb(0F) == System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.logb(-0F) == System.Single.NegativeInfinity);
        /// Assert.IsTrue(System.Single.IsNaN(math.logb(System.Single.NaN)));
        /// Assert.IsTrue(System.Single.IsNaN(math.logb(-System.Single.NaN)));
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.logb(1F) = 0F);
        /// Assert.IsTrue(math.logb(CSng(System.Math.E)) = 1F);
        /// Assert.IsTrue(math.logb(1024F) = 10F);
        /// Assert.IsTrue(math.logb(-2000F) = 10F);
        /// 
        /// Assert.IsTrue(math.logb(2F) = 1F);
        /// Assert.IsTrue(math.logb(CSng(Math.Pow(2F, 56F))) = 56F);
        /// Assert.IsTrue(math.logb(1.1F * CSng(Math.Pow(2F, -149F))) = -149F);
        /// Assert.IsTrue(math.logb(CSng(Math.Pow(2F, -150F))) = System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.logb(CSng(Math.Pow(2F, 128F))) = System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.logb(CSng(Math.Pow(2D, 127F))) = 127F);
        /// Assert.IsTrue(math.logb(2F * CSng(Math.Pow(2F, 102F))) = 103F);
        /// 
        /// Assert.IsTrue(math.logb(math.FLT_DENORM_MIN) = math.FLT_EXP_MIN - math.FLT_MANT_BITS);
        /// Assert.IsTrue(math.logb(math.FLT_DENORM_MAX) = math.FLT_EXP_MIN - 1);
        /// Assert.IsTrue(math.logb(math.FLT_MIN) = math.FLT_EXP_MIN);
        /// Assert.IsTrue(math.logb(math.FLT_MAX) = math.FLT_EXP_MAX);
        /// 
        /// Assert.IsTrue(math.logb(System.Single.PositiveInfinity) = System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.logb(System.Single.NegativeInfinity) = System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.logb(0F) = System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.logb(-0F) = System.Single.NegativeInfinity);
        /// Assert.IsTrue(System.Single.IsNaN(math.logb(System.Single.NaN)));
        /// Assert.IsTrue(System.Single.IsNaN(math.logb(-System.Single.NaN)));
        /// </code> 
        /// </example>
        public static float logb(float number)
        {
            int exp = math.ilogb(number);
            switch (exp)
            {
                case math.FP_ILOGB0:
                    return System.Single.NegativeInfinity;
                case math.FP_ILOGBNAN:
                    return System.Single.NaN;
                case math.INT_MAX:
                    return System.Single.PositiveInfinity;
            }
            return exp;
        }

        #endregion

        #region "scalbn"

        /// <summary>
        /// Scales the specified floating-point <paramref name="number"/> by 2^<paramref name="exponent"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <param name="exponent">The exponent of the power of two.</param>
        /// <returns>The value <c><paramref name="number"/> * 2^<paramref name="exponent"/></c>.</returns>
        /// <remarks>
        /// <para>
        /// Special values are treated as follows.
        /// </para>
        /// <list type="bullet">
        /// <item>If <paramref name="number"/> is <c>±0</c>, it is returned.</item>
        /// <item>If <paramref name="number"/> is infinite, it is returned.</item>
        /// <item>If <paramref name="exponent"/> is <c>0</c>, <paramref name="number"/> is returned.</item>
        /// <item>If <paramref name="number"/> is NaN, <see cref="System.Double.NaN"/> is returned.</item>
        /// </list>
        /// <para>
        /// The function <see cref="math.scalbn(double, int)"/> is equivalent to <see cref="math.ldexp(double, int)"/>.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/scalbn">scalbn</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.scalbn(0.8D, 4) == 12.8D);
        /// Assert.IsTrue(math.scalbn(-0.854375D, 5) == -27.34D);
        /// Assert.IsTrue(math.scalbn(1D, 0) == 1D);
        /// 
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN / 2D, 0) == math.DBL_MIN / 2D);
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN / 2D, 1) == math.DBL_MIN);
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS) == 2D * math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS - 1) == math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS) == math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS - 1) == math.DBL_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.scalbn(1D, System.Int32.MaxValue) == System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.scalbn(1D, System.Int32.MinValue) == 0D);
        /// Assert.IsTrue(math.scalbn(-1D, System.Int32.MaxValue) == System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.scalbn(-1D, System.Int32.MinValue) == -0D);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.scalbn(0.8D, 4) = 12.8D);
        /// Assert.IsTrue(math.scalbn(-0.854375D, 5) = -27.34D);
        /// Assert.IsTrue(math.scalbn(1D, 0) = 1D);
        /// 
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN / 2D, 0) = math.DBL_MIN / 2D);
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN / 2D, 1) = math.DBL_MIN);
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS) = 2D * math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS - 1) = math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS) = math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS - 1) = math.DBL_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.scalbn(1D, System.Int32.MaxValue) = System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.scalbn(1D, System.Int32.MinValue) = 0D);
        /// Assert.IsTrue(math.scalbn(-1D, System.Int32.MaxValue) = System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.scalbn(-1D, System.Int32.MinValue) = -0D);
        /// </code> 
        /// </example>
        public static double scalbn(double number, int exponent)
        {
            return math.scalbln(number, exponent);
        }

        /// <summary>
        /// Scales the specified floating-point <paramref name="number"/> by 2^<paramref name="exponent"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <param name="exponent">The exponent of the power of two.</param>
        /// <returns>The value <c><paramref name="number"/> * 2^<paramref name="exponent"/></c>.</returns>
        /// <remarks>
        /// <para>
        /// Special values are treated as follows.
        /// </para>
        /// <list type="bullet">
        /// <item>If <paramref name="number"/> is <c>±0</c>, it is returned.</item>
        /// <item>If <paramref name="number"/> is infinite, it is returned.</item>
        /// <item>If <paramref name="exponent"/> is <c>0</c>, <paramref name="number"/> is returned.</item>
        /// <item>If <paramref name="number"/> is NaN, <see cref="System.Single.NaN"/> is returned.</item>
        /// </list>
        /// <para>
        /// The function <see cref="math.scalbn(float, int)"/> is equivalent to <see cref="math.ldexp(float, int)"/>.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/scalbn">scalbn</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.scalbn(0.8F, 4) == 12.8F);
        /// Assert.IsTrue(math.scalbn(-0.854375F, 5) == -27.34F);
        /// Assert.IsTrue(math.scalbn(1F, 0) == 1F);
        /// 
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN / 2F, 0) == math.FLT_MIN / 2F);
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN / 2F, 1) == math.FLT_MIN);
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS) == 2F * math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS - 1) == math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS) == math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS - 1) == math.FLT_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.scalbn(1F, System.Int32.MaxValue) == System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.scalbn(1F, System.Int32.MinValue) == 0F);
        /// Assert.IsTrue(math.scalbn(-1F, System.Int32.MaxValue) == System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.scalbn(-1F, System.Int32.MinValue) == -0F);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.scalbn(0.8F, 4) = 12.8F);
        /// Assert.IsTrue(math.scalbn(-0.854375F, 5) = -27.34F);
        /// Assert.IsTrue(math.scalbn(1F, 0) = 1F);
        /// 
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN / 2F, 0) = math.FLT_MIN / 2F);
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN / 2F, 1) = math.FLT_MIN);
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS) = 2F * math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS - 1) = math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS) = math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbn(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS - 1) = math.FLT_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.scalbn(1F, System.Int32.MaxValue) = System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.scalbn(1F, System.Int32.MinValue) = 0F);
        /// Assert.IsTrue(math.scalbn(-1F, System.Int32.MaxValue) = System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.scalbn(-1F, System.Int32.MinValue) = -0F);
        /// </code> 
        /// </example>
        public static float scalbn(float number, int exponent)
        {
            return math.scalbln(number, exponent);
        }

        #endregion

        #region "scalbln"

        /// <summary>
        /// Scales the specified floating-point <paramref name="number"/> by 2^<paramref name="exponent"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <param name="exponent">The exponent of the power of two.</param>
        /// <returns>The value <c><paramref name="number"/> * 2^<paramref name="exponent"/></c>.</returns>
        /// <remarks>
        /// <para>
        /// Special values are treated as follows.
        /// </para>
        /// <list type="bullet">
        /// <item>If <paramref name="number"/> is <c>±0</c>, it is returned.</item>
        /// <item>If <paramref name="number"/> is infinite, it is returned.</item>
        /// <item>If <paramref name="exponent"/> is <c>0</c>, <paramref name="number"/> is returned.</item>
        /// <item>If <paramref name="number"/> is NaN, <see cref="System.Double.NaN"/> is returned.</item>
        /// </list>
        /// <para>
        /// The function <see cref="math.scalbln(double, long)"/> is equivalent to <see cref="math.ldexp(double, int)"/>.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/scalbn">scalbln</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.scalbln(0.8D, 4L) == 12.8D);
        /// Assert.IsTrue(math.scalbln(-0.854375D, 5L) == -27.34D);
        /// Assert.IsTrue(math.scalbln(1D, 0L) == 1D);
        /// 
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN / 2D, 0L) == math.DBL_MIN / 2D);
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN / 2D, 1L) == math.DBL_MIN);
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS) == 2D * math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS - 1) == math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS) == math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS - 1) == math.DBL_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.scalbln(1D, System.Int64.MaxValue) == System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.scalbln(1D, System.Int64.MinValue) == 0D);
        /// Assert.IsTrue(math.scalbln(-1D, System.Int64.MaxValue) == System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.scalbln(-1D, System.Int64.MinValue) == -0D);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.scalbln(0.8D, 4L) = 12.8D);
        /// Assert.IsTrue(math.scalbln(-0.854375D, 5L) = -27.34D);
        /// Assert.IsTrue(math.scalbln(1D, 0L) = 1D);
        /// 
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN / 2D, 0L) = math.DBL_MIN / 2D);
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN / 2D, 1L) = math.DBL_MIN);
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS) = 2D * math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN * 1.5D, -math.DBL_MANT_BITS - 1) = math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS) = math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.DBL_MIN * 1.25D, -math.DBL_MANT_BITS - 1) = math.DBL_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.scalbln(1D, System.Int64.MaxValue) = System.Double.PositiveInfinity);
        /// Assert.IsTrue(math.scalbln(1D, System.Int64.MinValue) = 0D);
        /// Assert.IsTrue(math.scalbln(-1D, System.Int64.MaxValue) = System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.scalbln(-1D, System.Int64.MinValue) = -0D);
        /// </code> 
        /// </example>
        public static double scalbln(double number, long exponent)
        {
            long bits = System.BitConverter.DoubleToInt64Bits(number);
            int exp = (int)((bits & math.DBL_EXP_MASK) >> math.DBL_MANT_BITS);
            // Check for infinity or NaN.
            if (exp == 0x7ff)
                return number;
            // Check for 0 or subnormal.
            if (exp == 0)
            {
                // Check for 0.
                if ((bits & math.DBL_MANT_MASK) == 0)
                    return number;
                // Subnormal, scale number so that it is in [1, 2).
                number *= System.BitConverter.Int64BitsToDouble(0x4350000000000000L); // 2^54
                bits = System.BitConverter.DoubleToInt64Bits(number);
                exp = (int)((bits & math.DBL_EXP_MASK) >> math.DBL_MANT_BITS) - 54;
            }
            // Check for underflow.
            if (exponent < -50000)
                return math.copysign(0D, number);
            // Check for overflow.
            if (exponent > 50000 || (long)exp + exponent > 0x7feL)
                return math.copysign(System.Double.PositiveInfinity, number);
            exp += (int)exponent;
            // Check for normal.
            if (exp > 0)
                return System.BitConverter.Int64BitsToDouble((bits & math.DBL_EXP_CLR_MASK) | ((long)exp << math.DBL_MANT_BITS));
            // Check for underflow.
            if (exp <= -54)
                return math.copysign(0D, number);
            // Subnormal.
            exp += 54;
            number = System.BitConverter.Int64BitsToDouble((bits & math.DBL_EXP_CLR_MASK) | ((long)exp << math.DBL_MANT_BITS));
            return number * System.BitConverter.Int64BitsToDouble(0x3c90000000000000L); // 2^-54
        }

        /// <summary>
        /// Scales the specified floating-point <paramref name="number"/> by 2^<paramref name="exponent"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <param name="exponent">The exponent of the power of two.</param>
        /// <returns>The value <c><paramref name="number"/> * 2^<paramref name="exponent"/></c>.</returns>
        /// <remarks>
        /// <para>
        /// Special values are treated as follows.
        /// </para>
        /// <list type="bullet">
        /// <item>If <paramref name="number"/> is <c>±0</c>, it is returned.</item>
        /// <item>If <paramref name="number"/> is infinite, it is returned.</item>
        /// <item>If <paramref name="exponent"/> is <c>0</c>, <paramref name="number"/> is returned.</item>
        /// <item>If <paramref name="number"/> is NaN, <see cref="System.Single.NaN"/> is returned.</item>
        /// </list>
        /// <para>
        /// The function <see cref="math.scalbln(float, long)"/> is equivalent to <see cref="math.ldexp(float, int)"/>.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/scalbn">scalbln</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.scalbln(0.8F, 4L) == 12.8F);
        /// Assert.IsTrue(math.scalbln(-0.854375F, 5L) == -27.34F);
        /// Assert.IsTrue(math.scalbln(1F, 0L) == 1F);
        /// 
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN / 2F, 0L) == math.FLT_MIN / 2F);
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN / 2F, 1L) == math.FLT_MIN);
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS) == 2F * math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS - 1) == math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS) == math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS - 1) == math.FLT_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.scalbln(1F, System.Int64.MaxValue) == System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.scalbln(1F, System.Int64.MinValue) == 0F);
        /// Assert.IsTrue(math.scalbln(-1F, System.Int64.MaxValue) == System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.scalbln(-1F, System.Int64.MinValue) == -0F);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.scalbln(0.8F, 4L) = 12.8F);
        /// Assert.IsTrue(math.scalbln(-0.854375F, 5L) = -27.34F);
        /// Assert.IsTrue(math.scalbln(1F, 0L) = 1F);
        /// 
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN / 2F, 0L) = math.FLT_MIN / 2F);
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN / 2F, 1L) = math.FLT_MIN);
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS) = 2F * math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN * 1.5F, -math.FLT_MANT_BITS - 1) = math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS) = math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.scalbln(math.FLT_MIN * 1.25F, -math.FLT_MANT_BITS - 1) = math.FLT_DENORM_MIN);
        /// 
        /// Assert.IsTrue(math.scalbln(1F, System.Int64.MaxValue) = System.Single.PositiveInfinity);
        /// Assert.IsTrue(math.scalbln(1F, System.Int64.MinValue) = 0F);
        /// Assert.IsTrue(math.scalbln(-1F, System.Int64.MaxValue) = System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.scalbln(-1F, System.Int64.MinValue) = -0F);
        /// </code> 
        /// </example>
        public static float scalbln(float number, long exponent)
        {
            int bits = math.SingleToInt32Bits(number);
            int exp = (bits & math.FLT_EXP_MASK) >> math.FLT_MANT_BITS;
            // Check for infinity or NaN.
            if (exp == 0xff)
                return number;
            // Check for 0 or subnormal.
            if (exp == 0)
            {
                // Check for 0.
                if ((bits & math.FLT_MANT_MASK) == 0)
                    return number;
                // Subnormal, scale number so that it is in [1, 2).
                number *= math.Int32BitsToSingle(0x4c000000); // 2^25
                bits = math.SingleToInt32Bits(number);
                exp = ((bits & math.FLT_EXP_MASK) >> math.FLT_MANT_BITS) - 25;
            }
            // Check for underflow.
            if (exponent < -50000)
                return math.copysign(0F, number);
            // Check for overflow.
            if (exponent > 50000 || exp + exponent > 0xfe)
                return math.copysign(System.Single.PositiveInfinity, number);
            exp += (int)exponent;
            // Check for normal.
            if (exp > 0)
                return math.Int32BitsToSingle((bits & math.FLT_EXP_CLR_MASK) | (exp << math.FLT_MANT_BITS));
            // Check for underflow.
            if (exp <= -25)
                return math.copysign(0F, number);
            // Subnormal.
            exp += 25;
            number = math.Int32BitsToSingle((bits & math.FLT_EXP_CLR_MASK) | (exp << math.FLT_MANT_BITS));
            return number * math.Int32BitsToSingle(0x33000000); // 2^-25
        }

        #endregion

        #endregion

        #region "Floating-point manipulation functions."

        #region "copysign"

        /// <summary>
        /// Copies the sign of <paramref name="number2"/> to <paramref name="number1"/>.
        /// </summary>
        /// <param name="number1">A floating-point number.</param>
        /// <param name="number2">A floating-point number.</param>
        /// <returns>The floating-point number whose absolute value is that of <paramref name="number1"/> with the sign of <paramref name="number2"/>.</returns>
        /// <remarks>
        /// <para>
        /// <see cref="math.copysign(double, double)"/> is the only portable way to manipulate the sign of a <see cref="System.Double.NaN"/> value (to examine
        /// the sign of a <see cref="System.Double.NaN"/>, <see cref="math.signbit(double)"/> may also be used). 
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/copysign">copysign</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.copysign(0D, -0D) == -0D);
        /// Assert.IsTrue(math.copysign(0D, -4D) == -0D);
        /// Assert.IsTrue(math.copysign(2D, -0D) == -2D);
        /// Assert.IsTrue(math.copysign(-2D, 0D) == 2D);
        /// Assert.IsTrue(math.copysign(System.Double.PositiveInfinity, -2D) == System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.copysign(2D, System.Double.NegativeInfinity) == -2D);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.copysign(0D, -0D) = -0D);
        /// Assert.IsTrue(math.copysign(0D, -4D) = -0D);
        /// Assert.IsTrue(math.copysign(2D, -0D) = -2D);
        /// Assert.IsTrue(math.copysign(-2D, 0D) = 2D);
        /// Assert.IsTrue(math.copysign(System.Double.PositiveInfinity, -2D) = System.Double.NegativeInfinity);
        /// Assert.IsTrue(math.copysign(2D, System.Double.NegativeInfinity) = -2D);
        /// </code> 
        /// </example>
        public static double copysign(double number1, double number2)
        {
            // If number1 is NaN, we have to store in it the opposite of the sign bit.
            long sign = (math.signbit(number2) == 1 ? math.DBL_SGN_MASK : 0L) ^ (System.Double.IsNaN(number1) ? math.DBL_SGN_MASK : 0L);
            return System.BitConverter.Int64BitsToDouble((System.BitConverter.DoubleToInt64Bits(number1) & math.DBL_SGN_CLR_MASK) | sign);
        }

        /// <summary>
        /// Copies the sign of <paramref name="number2"/> to <paramref name="number1"/>.
        /// </summary>
        /// <param name="number1">A floating-point number.</param>
        /// <param name="number2">A floating-point number.</param>
        /// <returns>The floating-point number whose absolute value is that of <paramref name="number1"/> with the sign of <paramref name="number2"/>.</returns>
        /// <remarks>
        /// <para>
        /// <see cref="math.copysign(float, float)"/> is the only portable way to manipulate the sign of a <see cref="System.Single.NaN"/> value (to examine
        /// the sign of a <see cref="System.Single.NaN"/>, <see cref="math.signbit(float)"/> may also be used). 
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/copysign">copysign</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.copysign(0F, -0F) == -0F);
        /// Assert.IsTrue(math.copysign(0F, -4F) == -0F);
        /// Assert.IsTrue(math.copysign(2F, -0F) == -2F);
        /// Assert.IsTrue(math.copysign(-2F, 0F) == 2F);
        /// Assert.IsTrue(math.copysign(System.Single.PositiveInfinity, -2F) == System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.copysign(2F, System.Single.NegativeInfinity) == -2F);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.copysign(0F, -0F) = -0F);
        /// Assert.IsTrue(math.copysign(0F, -4F) = -0F);
        /// Assert.IsTrue(math.copysign(2F, -0F) = -2F);
        /// Assert.IsTrue(math.copysign(-2F, 0F) = 2F);
        /// Assert.IsTrue(math.copysign(System.Single.PositiveInfinity, -2F) = System.Single.NegativeInfinity);
        /// Assert.IsTrue(math.copysign(2F, System.Single.NegativeInfinity) = -2F);
        /// </code> 
        /// </example>
        public static float copysign(float number1, float number2)
        {
            // If number1 is NaN, we have to store in it the opposite of the sign bit.
            int sign = (math.signbit(number2) == 1 ? math.FLT_SGN_MASK : 0) ^ (System.Double.IsNaN(number1) ? math.FLT_SGN_MASK : 0);
            return math.Int32BitsToSingle((math.SingleToInt32Bits(number1) & math.FLT_SGN_CLR_MASK) | sign);
        }

        #endregion

        #region "nextafter"

        /// <summary>
        /// Gets the floating-point number that is next after <paramref name="fromNumber"/> in the direction of <paramref name="towardNumber"/>.
        /// </summary>
        /// <param name="fromNumber">A floating-point number.</param>
        /// <param name="towardNumber">A floating-point number.</param>
        /// <returns>The floating-point number that is next after <paramref name="fromNumber"/> in the direction of <paramref name="towardNumber"/>.</returns>
        /// <remarks>
        /// <para>
        /// IEC 60559 recommends that <paramref name="fromNumber"/> be returned whenever <c><paramref name="fromNumber"/> == <paramref name="towardNumber"/></c>.
        /// These functions return <paramref name="towardNumber"/> instead, which makes the behavior around zero consistent: <c><see cref="math.nextafter(double, double)">nextafter</see>(-0.0, +0.0)</c>
        /// returns <c>+0.0</c> and <c><see cref="math.nextafter(double, double)">nextafter</see>(+0.0, -0.0)</c> returns <c>–0.0</c>.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/nextafter">nextafter</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.nextafter(0D, 0D) == 0D);
        /// Assert.IsTrue(math.nextafter(-0D, 0D) == 0D;
        /// Assert.IsTrue(math.nextafter(0D, -0D) == -0D);
        /// 
        /// Assert.IsTrue(math.nextafter(math.DBL_MIN, 0D) == math.DBL_DENORM_MAX);
        /// Assert.IsTrue(math.nextafter(math.DBL_DENORM_MIN, 0D) == 0D);
        /// Assert.IsTrue(math.nextafter(math.DBL_MIN, -0D) == math.DBL_DENORM_MAX);
        /// Assert.IsTrue(math.nextafter(math.DBL_DENORM_MIN, -0D) == 0D);
        /// 
        /// Assert.IsTrue(math.nextafter(0D, System.Double.PositiveInfinity) == math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.nextafter(-0D, System.Double.NegativeInfinity) == -math.DBL_DENORM_MIN);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.nextafter(0D, 0D) = 0D);
        /// Assert.IsTrue(math.nextafter(-0D, 0D) = 0D);
        /// Assert.IsTrue(math.nextafter(0D, -0D) = -0D);
        /// 
        /// Assert.IsTrue(math.nextafter(math.DBL_MIN, 0D) = math.DBL_DENORM_MAX);
        /// Assert.IsTrue(math.nextafter(math.DBL_DENORM_MIN, 0D) = 0D);
        /// Assert.IsTrue(math.nextafter(math.DBL_MIN, -0D) = math.DBL_DENORM_MAX);
        /// Assert.IsTrue(math.nextafter(math.DBL_DENORM_MIN, -0D) = 0D);
        /// 
        /// Assert.IsTrue(math.nextafter(0D, System.Double.PositiveInfinity) = math.DBL_DENORM_MIN);
        /// Assert.IsTrue(math.nextafter(-0D, System.Double.NegativeInfinity) = -math.DBL_DENORM_MIN);
        /// </code> 
        /// </example>
        public static double nextafter(double fromNumber, double towardNumber)
        {
            // If either fromNumber or towardNumber is NaN, return NaN.
            if (System.Double.IsNaN(towardNumber) || System.Double.IsNaN(fromNumber))
            {
                return System.Double.NaN;
            }
            // If no direction.
            if (fromNumber == towardNumber)
            {
                return towardNumber;
            }
            // If fromNumber is zero, return smallest subnormal.
            if (fromNumber == 0)
            {
                return (towardNumber > 0) ? System.Double.Epsilon : -System.Double.Epsilon;
            }
            // All other cases are handled by incrementing or decrementing the bits value.
            // Transitions to infinity, to subnormal, and to zero are all taken care of this way.
            long bits = System.BitConverter.DoubleToInt64Bits(fromNumber);
            // A xor here avoids nesting conditionals. We have to increment if fromValue lies between 0 and toValue.
            if ((fromNumber > 0) ^ (fromNumber > towardNumber))
            {
                bits += 1;
            }
            else
            {
                bits -= 1;
            }
            return System.BitConverter.Int64BitsToDouble(bits);
        }

        /// <summary>
        /// Gets the floating-point number that is next after <paramref name="fromNumber"/> in the direction of <paramref name="towardNumber"/>.
        /// </summary>
        /// <param name="fromNumber">A floating-point number.</param>
        /// <param name="towardNumber">A floating-point number.</param>
        /// <returns>The floating-point number that is next after <paramref name="fromNumber"/> in the direction of <paramref name="towardNumber"/>.</returns>
        /// <remarks>
        /// <para>
        /// IEC 60559 recommends that <paramref name="fromNumber"/> be returned whenever <c><paramref name="fromNumber"/> == <paramref name="towardNumber"/></c>.
        /// These functions return <paramref name="towardNumber"/> instead, which makes the behavior around zero consistent: <c><see cref="math.nextafter(float, float)">nextafter</see>(-0.0, +0.0)</c>
        /// returns <c>+0.0</c> and <c><see cref="math.nextafter(float, float)">nextafter</see>(+0.0, -0.0)</c> returns <c>–0.0</c>.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/nextafter">nextafter</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.nextafter(0F, 0F) == 0F);
        /// Assert.IsTrue(math.nextafter(-0F, 0F) == 0F;
        /// Assert.IsTrue(math.nextafter(0F, -0F) == -0F);
        /// 
        /// Assert.IsTrue(math.nextafter(math.FLT_MIN, 0D) == math.FLT_DENORM_MAX);
        /// Assert.IsTrue(math.nextafter(math.FLT_DENORM_MIN, 0F) == 0F);
        /// Assert.IsTrue(math.nextafter(math.FLT_MIN, -0F) == math.FLT_DENORM_MAX);
        /// Assert.IsTrue(math.nextafter(math.FLT_DENORM_MIN, -0F) == 0F);
        /// 
        /// Assert.IsTrue(math.nextafter(0F, System.Single.PositiveInfinity) == math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.nextafter(-0F, System.Single.NegativeInfinity) == -math.FLT_DENORM_MIN);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.nextafter(0F, 0F) = 0F);
        /// Assert.IsTrue(math.nextafter(-0F, 0F) = 0F);
        /// Assert.IsTrue(math.nextafter(0F, -0F) = -0F);
        /// 
        /// Assert.IsTrue(math.nextafter(math.FLT_MIN, 0F) = math.FLT_DENORM_MAX);
        /// Assert.IsTrue(math.nextafter(math.FLT_DENORM_MIN, 0F) = 0F);
        /// Assert.IsTrue(math.nextafter(math.FLT_MIN, -0F) = math.FLT_DENORM_MAX);
        /// Assert.IsTrue(math.nextafter(math.FLT_DENORM_MIN, -0F) = 0F);
        /// 
        /// Assert.IsTrue(math.nextafter(0F, System.Single.PositiveInfinity) = math.FLT_DENORM_MIN);
        /// Assert.IsTrue(math.nextafter(-0F, System.Single.NegativeInfinity) = -math.FLT_DENORM_MIN);
        /// </code> 
        /// </example>
        public static float nextafter(float fromNumber, float towardNumber)
        {
            // If either fromNumber or towardNumber is NaN, return NaN.
            if (System.Single.IsNaN(towardNumber) || System.Single.IsNaN(fromNumber))
            {
                return System.Single.NaN;
            }
            // If no direction or if fromNumber is infinity or is not a number, return fromNumber.
            if (fromNumber == towardNumber)
            {
                return towardNumber;
            }
            // If fromNumber is zero, return smallest subnormal.
            if (fromNumber == 0)
            {
                return (towardNumber > 0) ? System.Single.Epsilon : -System.Single.Epsilon;
            }
            // All other cases are handled by incrementing or decrementing the bits value.
            // Transitions to infinity, to subnormal, and to zero are all taken care of this way.
            int bits = SingleToInt32Bits(fromNumber);
            // A xor here avoids nesting conditionals. We have to increment if fromValue lies between 0 and toValue.
            if ((fromNumber > 0) ^ (fromNumber > towardNumber))
            {
                bits += 1;
            }
            else
            {
                bits -= 1;
            }
            return Int32BitsToSingle(bits);
        }

        #endregion

        #region "nexttoward"

        /// <summary>
        /// Gets the floating-point number that is next after <paramref name="fromNumber"/> in the direction of <paramref name="towardNumber"/>.
        /// </summary>
        /// <param name="fromNumber">A floating-point number.</param>
        /// <param name="towardNumber">A floating-point number.</param>
        /// <returns>The floating-point number that is next after <paramref name="fromNumber"/> in the direction of <paramref name="towardNumber"/>.</returns>
        /// <remarks>
        /// <para>
        /// IEC 60559 recommends that <paramref name="fromNumber"/> be returned whenever <c><paramref name="fromNumber"/> == <paramref name="towardNumber"/></c>.
        /// These functions return <paramref name="towardNumber"/> instead, which makes the behavior around zero consistent: <c><see cref="math.nexttoward(double, double)">nexttoward</see>(-0.0, +0.0)</c>
        /// returns <c>+0.0</c> and <c><see cref="math.nexttoward(double, double)">nexttoward</see>(+0.0, -0.0)</c> returns <c>–0.0</c>.
        /// </para>
        /// <para>
        /// The <see cref="math.nexttoward(double, double)"/> function is equivalent to the <see cref="math.nextafter(double, double)"/> function.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/nextafter">nexttoward</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static double nexttoward(double fromNumber, double towardNumber)
        {
            return math.nextafter(fromNumber, towardNumber);
        }

        /// <summary>
        /// Gets the floating-point number that is next after <paramref name="fromNumber"/> in the direction of <paramref name="towardNumber"/>.
        /// </summary>
        /// <param name="fromNumber">A floating-point number.</param>
        /// <param name="towardNumber">A floating-point number.</param>
        /// <returns>The floating-point number that is next after <paramref name="fromNumber"/> in the direction of <paramref name="towardNumber"/>.</returns>
        /// <remarks>
        /// <para>
        /// IEC 60559 recommends that <paramref name="fromNumber"/> be returned whenever <c><paramref name="fromNumber"/> == <paramref name="towardNumber"/></c>.
        /// These functions return <paramref name="towardNumber"/> instead, which makes the behavior around zero consistent: <c><see cref="math.nexttoward(float, float)">nexttoward</see>(-0.0, +0.0)</c>
        /// returns <c>+0.0</c> and <c><see cref="math.nexttoward(float, float)">nexttoward</see>(+0.0, -0.0)</c> returns <c>–0.0</c>.
        /// </para>
        /// <para>
        /// The <see cref="math.nexttoward(float, float)"/> function is equivalent to the <see cref="math.nextafter(float, float)"/> function.
        /// </para>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/nextafter">nexttoward</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static float nexttoward(float fromNumber, float towardNumber)
        {
            return math.nextafter(fromNumber, towardNumber);
        }

        #endregion

        #region "exponent"

        /// <summary>
        /// Gets the exponent bits of the specified floating-point <paramref name="number"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The exponent bits of the specified floating-point <paramref name="number"/>; i.e. the biased exponent.</returns>
        /// <remarks>
        /// <list type="table">
        ///     <listheader>
        ///        <term><paramref name="number"/></term> 
        ///        <description>Biased Exponent</description> 
        ///        <description>Unbiased Exponent</description> 
        ///     </listheader>
        ///     <item>
        ///         <term><c>±<see cref="System.Double.NaN"/></c></term>
        ///         <description><c>2047</c> (<c><see cref="math.DBL_EXP_MAX"/> + 1 + <see cref="math.DBL_EXP_BIAS"/></c>)</description>
        ///         <description>N/A</description>
        ///     </item>
        ///     <item>
        ///         <term><c><see cref="System.Double.PositiveInfinity"/></c></term>
        ///         <description><c>2047</c> (<c><see cref="math.DBL_EXP_MAX"/> + 1 + <see cref="math.DBL_EXP_BIAS"/></c>)</description>
        ///         <description>N/A</description>
        ///     </item>
        ///     <item>
        ///         <term><c><see cref="System.Double.NegativeInfinity"/></c></term>
        ///         <description><c>2047</c> (<c><see cref="math.DBL_EXP_MAX"/> + 1 + <see cref="math.DBL_EXP_BIAS"/></c>)</description>
        ///         <description>N/A</description>
        ///     </item>
        ///     <item>
        ///         <term><c>±<see cref="math.DBL_MAX"/></c></term>
        ///         <description><c>2046</c> (<c><see cref="math.DBL_EXP_MAX"/> + <see cref="math.DBL_EXP_BIAS"/></c>)</description>
        ///         <description><c>1023</c> (<c><see cref="math.DBL_EXP_MAX"/></c>)</description>
        ///     </item>
        ///     <item>
        ///         <term><c>±<see cref="math.DBL_MIN"/></c></term>
        ///         <description><c>1</c> (<c><see cref="math.DBL_EXP_MIN"/> + <see cref="math.DBL_EXP_BIAS"/></c>)</description>
        ///         <description><c>-1022</c> (<c><see cref="math.DBL_EXP_MIN"/></c>)</description>
        ///     </item>
        ///     <item>
        ///         <term><c>±<see cref="math.DBL_DENORM_MAX"/></c></term>
        ///         <description><c>0</c></description>
        ///         <description><c>0</c></description>
        ///     </item>
        ///     <item>
        ///         <term><c>±<see cref="math.DBL_DENORM_MIN"/></c></term>
        ///         <description><c>0</c></description>
        ///         <description><c>0</c></description>
        ///     </item>
        ///     <item>
        ///         <term><c>±0</c></term>
        ///         <description><c>0</c></description>
        ///         <description><c>0</c></description>
        ///     </item>
        /// </list>
        /// </remarks>
        public static int exponent(double number)
        {
            return System.Convert.ToInt32((System.BitConverter.DoubleToInt64Bits(number) & math.DBL_EXP_MASK) >> math.DBL_MANT_BITS);
        }

        /// <summary>
        /// Gets the exponent bits of the specified floating-point <paramref name="number"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The exponent bits of the specified floating-point <paramref name="number"/>; i.e. the biased exponent.</returns>
        /// <remarks>
        /// <list type="table">
        ///     <listheader>
        ///        <term><paramref name="number"/></term> 
        ///        <description>Biased Exponent</description> 
        ///        <description>Unbiased Exponent</description> 
        ///     </listheader>
        ///     <item>
        ///         <term><c>±<see cref="System.Single.NaN"/></c></term>
        ///         <description><c>255</c> (<c><see cref="math.FLT_EXP_MAX"/> + 1 + <see cref="math.FLT_EXP_BIAS"/></c>)</description>
        ///         <description>N/A</description>
        ///     </item>
        ///     <item>
        ///         <term><c><see cref="System.Single.PositiveInfinity"/></c></term>
        ///         <description><c>255</c> (<c><see cref="math.FLT_EXP_MAX"/> + 1 + <see cref="math.FLT_EXP_BIAS"/></c>)</description>
        ///         <description>N/A</description>
        ///     </item>
        ///     <item>
        ///         <term><c><see cref="System.Single.NegativeInfinity"/></c></term>
        ///         <description><c>255</c> (<c><see cref="math.FLT_EXP_MAX"/> + 1 + <see cref="math.FLT_EXP_BIAS"/></c>)</description>
        ///         <description>N/A</description>
        ///     </item>
        ///     <item>
        ///         <term><c>±<see cref="math.FLT_MAX"/></c></term>
        ///         <description><c>255</c> (<c><see cref="math.FLT_EXP_MAX"/> + 1 + <see cref="math.FLT_EXP_BIAS"/></c>)</description>
        ///         <description><c>128</c> (<c><see cref="math.FLT_EXP_MAX"/> + 1</c>)</description>
        ///     </item>
        ///     <item>
        ///         <term><c>±<see cref="math.FLT_MIN"/></c></term>
        ///         <description><c>255</c> (<c><see cref="math.FLT_EXP_MAX"/> + 1 + <see cref="math.FLT_EXP_BIAS"/></c>)</description>
        ///         <description><c>-127</c> (<c><see cref="math.FLT_EXP_MAX"/> + 1</c>)</description>
        ///     </item>
        ///     <item>
        ///         <term><c>±<see cref="math.FLT_DENORM_MAX"/></c></term>
        ///         <description><c>0</c></description>
        ///         <description><c>0</c></description>
        ///     </item>
        ///     <item>
        ///         <term><c>±<see cref="math.FLT_DENORM_MIN"/></c></term>
        ///         <description><c>0</c></description>
        ///         <description><c>0</c></description>
        ///     </item>
        ///     <item>
        ///         <term><c>±0</c></term>
        ///         <description><c>0</c></description>
        ///         <description><c>0</c></description>
        ///     </item>
        /// </list>
        /// </remarks>
        public static int exponent(float number)
        {
            return System.Convert.ToInt32((math.SingleToInt32Bits(number) & math.FLT_EXP_MASK) >> math.FLT_MANT_BITS);
        }

        #endregion

        #region "mantissa"

        /// <summary>
        /// Gets the mantissa bits of the specified floating-point <paramref name="number"/> without the implicit leading <c>1</c> bit.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The mantissa bits of the specified floating-point <paramref name="number"/> without the implicit leading <c>1</c> bit.</returns>
        /// <remarks></remarks>
        public static long mantissa(double number)
        {
            return System.BitConverter.DoubleToInt64Bits(number) & math.DBL_MANT_MASK;
        }

        /// <summary>
        /// Gets the mantissa bits of the specified floating-point <paramref name="number"/> without the implicit leading <c>1</c> bit.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The mantissa bits of the specified floating-point <paramref name="number"/> without the implicit leading <c>1</c> bit.</returns>
        /// <remarks></remarks>
        public static int mantissa(float number)
        {
            return math.SingleToInt32Bits(number) & math.FLT_MANT_MASK;
        }

        #endregion

        #region "significand"

        /// <summary>
        /// Gets the significand of the specified floating-point <paramref name="number"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The significand of the specified floating-point <paramref name="number"/>, or <paramref name="number"/> if it not normal or subnormal.</returns>
        /// <remarks>
        /// <para>
        /// The significand is a number in the interval <c>[1, 2)</c> so that 
        /// <c><paramref name="number"/> = <see cref="math.significand(double)">significand</see>(<paramref name="number"/>) * 2^<see cref="math.logb(double)">logb</see>(<paramref name="number"/>)</c>.
        /// If <paramref name="number"/> is subnormal, it is normalized so that the significand falls in the interval <c>[1, 2)</c>.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.significand(0D) == 0D);
        /// Assert.IsTrue(math.significand(-0D) == -0D);
        /// Assert.IsTrue(math.significand(1D) == 1D);
        /// Assert.IsTrue(math.significand(4D) == 1D);
        /// Assert.IsTrue(math.significand(6D) == 1.5D);
        /// Assert.IsTrue(math.significand(7D) == 1.75D);
        /// Assert.IsTrue(math.significand(8D) == 1D);
        /// Assert.IsTrue(math.significand(math.DBL_DENORM_MIN) == 1D);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.significand(0D) = 0D);
        /// Assert.IsTrue(math.significand(-0D) = -0D);
        /// Assert.IsTrue(math.significand(1D) = 1D);
        /// Assert.IsTrue(math.significand(4D) = 1D);
        /// Assert.IsTrue(math.significand(6D) = 1.5D);
        /// Assert.IsTrue(math.significand(7D) = 1.75D);
        /// Assert.IsTrue(math.significand(8D) = 1D);
        /// Assert.IsTrue(math.significand(math.DBL_DENORM_MIN) = 1D);
        /// </code> 
        /// </example>
        public static double significand(double number)
        {
            // If not-a-numbner or infinity, simply return number.
            if (System.Double.IsNaN(number) || System.Double.IsInfinity(number))
                return number;
            // Get the mantissa bits.
            long mantissa = math.mantissa(number);
            // If the unbiased exponent is 0, we have either 0 or a subnormal number.
            if (math.exponent(number) == 0)
            {
                // If number is zero, return zero.
                if (mantissa == 0L)
                    return number;
                // Otherwise, shift the mantissa to the left until its first 1-bit makes
                // the mantissa larger than or equal to the mantissa mask, and reset the
                // the leading 1 bit. This yields a "normalized" number.
                while (mantissa < math.DBL_MANT_MASK)
                {
                    mantissa <<= 1;
                }
                mantissa = mantissa & math.DBL_MANT_MASK;
            }
            // Build new double with exponent 0 and the normalized mantissa.
            return System.BitConverter.Int64BitsToDouble((System.Convert.ToInt64(math.DBL_EXP_BIAS) << math.DBL_MANT_BITS) | mantissa | (math.signbit(number) == 1 ? math.DBL_SGN_MASK : 0L));
        }

        /// <summary>
        /// Gets the significand of the specified floating-point <paramref name="number"/>.
        /// </summary>
        /// <param name="number">A floating-point number.</param>
        /// <returns>The significand of the specified floating-point <paramref name="number"/>, or <paramref name="number"/> if it not normal or subnormal.</returns>
        /// <remarks>
        /// <para>
        /// The significand is a number in the interval <c>[1, 2)</c> so that 
        /// <c><paramref name="number"/> = <see cref="math.significand(float)"/>(<paramref name="number"/>) * 2^<see cref="math.logb(float)"/>(<paramref name="number"/>)</c>.
        /// If <paramref name="number"/> is subnormal, it is normalized so that the significand falls in the interval <c>[1, 2)</c>.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code language="C#">
        /// Assert.IsTrue(math.significand(0F) == 0F);
        /// Assert.IsTrue(math.significand(-0F) == -0F);
        /// Assert.IsTrue(math.significand(1F) == 1F);
        /// Assert.IsTrue(math.significand(4F) == 1F);
        /// Assert.IsTrue(math.significand(6F) == 1.5F);
        /// Assert.IsTrue(math.significand(7F) == 1.75F);
        /// Assert.IsTrue(math.significand(8F) == 1F);
        /// Assert.IsTrue(math.significand(math.FLT_DENORM_MIN) == 1F);
        /// </code> 
        /// <code language="VB.NET">
        /// Assert.IsTrue(math.significand(0F) = 0F);
        /// Assert.IsTrue(math.significand(-0F) = -0F);
        /// Assert.IsTrue(math.significand(1F) = 1F);
        /// Assert.IsTrue(math.significand(4F) = 1F);
        /// Assert.IsTrue(math.significand(6F) = 1.5F);
        /// Assert.IsTrue(math.significand(7F) = 1.75F);
        /// Assert.IsTrue(math.significand(8F) = 1F);
        /// Assert.IsTrue(math.significand(math.FLT_DENORM_MIN) = 1F);
        /// </code> 
        /// </example>
        public static float significand(float number)
        {
            // If not-a-numbner or infinity, simply return number.
            if (System.Single.IsNaN(number) || System.Single.IsInfinity(number))
                return number;
            // Get the mantissa bits.
            int mantissa = math.mantissa(number);
            // If the unbiased exponent is 0, we have either 0 or a subnormal number.
            if (math.exponent(number) == 0)
            {
                // If number is zero, return zero.
                if (mantissa == 0F)
                    return number;
                // Otherwise, shift the mantissa to the left until its first 1-bit makes
                // the mantissa larger than or equal to the mantissa mask, and reset the
                // the leading 1 bit. This yields a "normalized" number.
                while (mantissa < math.FLT_MANT_MASK)
                {
                    mantissa <<= 1;
                }
                mantissa = mantissa & math.FLT_MANT_MASK;
            }
            // Build new float with exponent 0 and the normalized mantissa.
            return math.Int32BitsToSingle((System.Convert.ToInt32(math.FLT_EXP_BIAS) << math.FLT_MANT_BITS) | mantissa | (math.signbit(number) == 1 ? math.FLT_SGN_MASK : 0));
        }

        #endregion

        #endregion

        #region "Comparison functions."

        #region "isunordered"

        /// <summary>
        /// Gets a value that indicates whether two floating-point numbers are unordered.
        /// </summary>
        /// <param name="number1">A floating-point number.</param>
        /// <param name="number2">A floating-point number.</param>
        /// <returns><c>true</c> if either <paramref name="number1"/> or <paramref name="number1"/> is <see cref="System.Double.NaN"/>, <c>false</c> otherwise.</returns>
        /// <remarks>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/isunordered">isunordered</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static bool isunordered(double number1, double number2)
        {
            return System.Double.IsNaN(number1) || System.Double.IsNaN(number2);
        }

        /// <summary>
        /// Gets a value that indicates whether two floating-point numbers are unordered.
        /// </summary>
        /// <param name="number1">A floating-point number.</param>
        /// <param name="number2">A floating-point number.</param>
        /// <returns><c>true</c> if either <paramref name="number1"/> or <paramref name="number1"/> is <see cref="System.Single.NaN"/>, <c>false</c> otherwise.</returns>
        /// <remarks>
        /// <para>
        /// See <a href="http://en.cppreference.com/w/c/numeric/math/isunordered">isunordered</a> in the C standard documentation.
        /// </para>
        /// </remarks>
        public static bool isunordered(float number1, float number2)
        {
            return System.Single.IsNaN(number1) || System.Single.IsNaN(number2);
        }

        #endregion

        #endregion

        #region "Miscellaneous functions."

        /// <summary>
        /// Converts the specified single-precision floating point number to a 32-bit signed integer.
        /// </summary>
        /// <param name="value">The number to convert.</param>
        /// <returns>A 32-bit signed integer whose value is equivalent to <paramref name="value"/>.</returns>
        public static unsafe int SingleToInt32Bits(float value)
        {
            return *((int*)&value);
        }

        /// <summary>
        /// Converts the specified 32-bit signed integer to a single-precision floating point number.
        /// </summary>
        /// <param name="value">The number to convert.</param>
        /// <returns>A double-precision floating point number whose value is equivalent to <paramref name="value"/>.</returns>
        public static unsafe float Int32BitsToSingle(int value)
        {
            return *((float*)&value);
        }

        private static int _leadingZeroesCount(int x)
        {
            int y;
            int n = 32;
            y = x >> 16; if (y != 0) { n = n - 16; x = y; }
            y = x >> 8; if (y != 0) { n = n - 8; x = y; }
            y = x >> 4; if (y != 0) { n = n - 4; x = y; }
            y = x >> 2; if (y != 0) { n = n - 2; x = y; }
            y = x >> 1; if (y != 0) return n - 2;
            return n - x;
        }

        private static int _leadingZeroesCount(long x)
        {
            long y;
            int n = 64;
            y = x >> 32; if (y != 0) { n = n - 32; x = y; }
            y = x >> 16; if (y != 0) { n = n - 16; x = y; }
            y = x >> 8; if (y != 0) { n = n - 8; x = y; }
            y = x >> 4; if (y != 0) { n = n - 4; x = y; }
            y = x >> 2; if (y != 0) { n = n - 2; x = y; }
            y = x >> 1; if (y != 0) return n - 2;
            return n - (int)x;
        }

        #endregion
    }
}
