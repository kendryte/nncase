using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace Nncase.IO
{
    public class BitWriter
    {
        public readonly StringBuilder BinString;

        /// <summary>
        /// number of bit
        /// </summary>
        public int Length
        {
            get
            {
                return BinString.Length;
            }
        }

        /// <summary>
        /// create a BitWriter
        /// </summary>
        public BitWriter()
        {
            BinString = new StringBuilder();
        }

        /// <summary>
        /// create a BitWriter
        /// </summary>
        /// <param name="bitLength">bit length</param>
        public BitWriter(int bitLength)
        {
            var add = 8 - bitLength % 8;
            BinString = new StringBuilder(bitLength + add);
        }

        /// <summary>
        /// write byte to bit stream
        /// </summary>
        /// <param name="b">byte value</param>
        /// <param name="bitLength">length in the bit stream</param>
        public void Write(byte b, int bitLength = 8)
        {
            var bin = Convert.ToString(b, 2);
            AppendBinString(bin, bitLength);
        }

        /// <summary>
        /// write int to bit stream
        /// </summary>
        /// <param name="i">int value</param>
        /// <param name="bitLength">length in the bit stream</param>
        public void Write(int i, int bitLength = 16)
        {
            var bin = Convert.ToString(i, 2);
            AppendBinString(bin, bitLength);
        }

        /// <summary>
        /// write int to bit stream
        /// </summary>
        /// <param name="i">int value</param>
        /// <param name="bitLength">length in the bit stream</param>
        public void Write(long i, int bitLength = 64)
        {
            var bin = Convert.ToString(i, 2);
            AppendBinString(bin, bitLength);
        }


        /// <summary>
        /// write char to bit stream
        /// </summary>
        /// <param name="c">char value</param>
        /// <param name="bitLength">length in the bit  stream</param>
        public void Write(char c, int bitLength = 7)
        {
            var b = Convert.ToByte(c);
            var bin = Convert.ToString(b, 2);
            AppendBinString(bin, bitLength);
        }

        /// <summary>
        /// wirte bool value to bit stream
        /// </summary>
        /// <param name="b">bool value</param>
        /// <param name="bitLength">length int the bit stream</param>
        public void Write(bool b, int bitLength = 1)
        {
            var bin = b ? "1" : "0";
            AppendBinString(bin, bitLength);
        }

        /// <summary>
        /// write binary string to bit stream
        /// </summary>
        /// <param name="bin">binary string</param>
        public void WriteBinaryString(string bin)
        {
            BinString.Append(bin);
        }

        /// <summary>
        /// get bytes in the writer stream, 8 bit align
        /// </summary>
        /// <returns></returns>
        public byte[] GetBytes()
        {
            var bin = GetBinString();
            var len = bin.Length / 8;
            var result = new byte[len];

            for (int i = 0; i < len; i++)
            {
                var bits = bin.Substring(i * 8, 8);
                result[i] = Convert.ToByte(bits, 2);
            }

            return result;
        }

        /// <summary>
        /// get binary string in the writer stream, 8 bit align
        /// </summary>
        /// <returns></returns>
        public string GetBinString()
        {
            var add = GetAdditionalBits();
            return BinString.ToString() + add;
        }

        private void AppendBinString(string bin, int bitLength)
        {
            if (bin.Length > bitLength)
                throw new Exception("len is too short");
            var add = bitLength - bin.Length;
            for (int i = 0; i < add; i++)
            {
                BinString.Append('0');
            }
            BinString.Append(bin);
        }

        private string GetAdditionalBits()
        {
            var add = 8 - BinString.Length % 8;
            if (add == 0) return string.Empty;

            var result = new StringBuilder(add);
            for (int i = 0; i < add; i++)
            {
                result.Append('0');
            }
            return result.ToString();
        }

        /// <summary>
        /// get the binary bytes.
        /// </summary>
        /// <returns> the bytes. </returns>
        public Byte[] ToBytes()
        {
            var binString = GetBinString();
            int numOfBytes = binString.Length / 8;
            var bytes = new byte[numOfBytes];
            for (int i = 0; i < numOfBytes; ++i)
            {
                bytes[i] = Convert.ToByte(binString.Substring(8 * i, 8), 2);
            }
            return bytes;
        }
    }
}