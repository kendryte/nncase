using System.IO;
namespace Nncase.IO
{

    //
    // 摘要:
    //     Writes primitive types in binary to a stream and supports writing strings in
    //     a specific encoding.
    public class BitWriter : IAsyncDisposable, IDisposable
    {
        /// <summary>
        /// Holds the underlying stream.
        /// </summary>
        protected Stream OutStream;

        //
        // 摘要:
        //     Gets the underlying stream of the System.IO.BinaryWriter.
        //
        // 返回结果:
        //     The underlying stream associated with the BinaryWriter.
        public virtual Stream BaseStream
        {
            get
            {
                throw null;
            }
        }

        //
        // 摘要:
        //     Initializes a new instance of the System.IO.BinaryWriter class that writes to a stream.
        protected BitWriter()
        {
            OutStream= new MemoryStream();
        }

        //
        // 摘要:
        //     Initializes a new instance of the System.IO.BinaryWriter class based on the specified
        //     stream and using UTF-8 encoding.
        //
        // 参数:
        //   output:
        //     The output stream.
        //
        // 异常:
        //   T:System.ArgumentException:
        //     The stream does not support writing or is already closed.
        //
        //   T:System.ArgumentNullException:
        //     output is null.
        public BitWriter(Stream output)
        {
            OutStream = output;
        }

        //
        // 摘要:
        //     Initializes a new instance of the System.IO.BinaryWriter class based on the specified
        //     stream and character encoding.
        //
        // 参数:
        //   output:
        //     The output stream.
        //
        //   encoding:
        //     The character encoding to use.
        //
        // 异常:
        //   T:System.ArgumentException:
        //     The stream does not support writing or is already closed.
        //
        //   T:System.ArgumentNullException:
        //     output or encoding is null.
        public BitWriter(Stream output, Encoding encoding)
        {
            throw new NotImplementedException();
        }

        //
        // 摘要:
        //     Initializes a new instance of the System.IO.BinaryWriter class based on the specified
        //     stream and character encoding, and optionally leaves the stream open.
        //
        // 参数:
        //   output:
        //     The output stream.
        //
        //   encoding:
        //     The character encoding to use.
        //
        //   leaveOpen:
        //     true to leave the stream open after the System.IO.BinaryWriter object is disposed;
        //     otherwise, false.
        //
        // 异常:
        //   T:System.ArgumentException:
        //     The stream does not support writing or is already closed.
        //
        //   T:System.ArgumentNullException:
        //     output or encoding is null.
        public BitWriter(Stream output, Encoding encoding, bool leaveOpen)
        {
            throw new NotSupportedException();
        }

        //
        // 摘要:
        //     Closes the current System.IO.BinaryWriter and the underlying stream.
        public virtual void Close()
        {
        }

        //
        // 摘要:
        //     Releases all resources used by the current instance of the System.IO.BinaryWriter
        //     class.
        public void Dispose()
        {
        }

        //
        // 摘要:
        //     Releases the unmanaged resources used by the System.IO.BinaryWriter and optionally
        //     releases the managed resources.
        //
        // 参数:
        //   disposing:
        //     true to release both managed and unmanaged resources; false to release only unmanaged
        //     resources.
        protected virtual void Dispose(bool disposing)
        {
        }

        //
        // 摘要:
        //     Asynchronously releases all resources used by the current instance of the System.IO.BinaryWriter
        //     class.
        //
        // 返回结果:
        //     A task that represents the asynchronous dispose operation.
        public virtual ValueTask DisposeAsync()
        {
            throw null;
        }

        //
        // 摘要:
        //     Clears all buffers for the current writer and causes any buffered data to be
        //     written to the underlying device.
        public virtual void Flush()
        {
        }

        //
        // 摘要:
        //     Sets the position within the current stream.
        //
        // 参数:
        //   offset:
        //     A byte offset relative to origin.
        //
        //   origin:
        //     A field of System.IO.SeekOrigin indicating the reference point from which the
        //     new position is to be obtained.
        //
        // 返回结果:
        //     The position with the current stream.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     The file pointer was moved to an invalid location.
        //
        //   T:System.ArgumentException:
        //     The System.IO.SeekOrigin value is invalid.
        public virtual long Seek(int offset, SeekOrigin origin)
        {
            throw null;
        }

        //
        // 摘要:
        //     Writes a one-byte Boolean value to the current stream, with 0 representing false
        //     and 1 representing true.
        //
        // 参数:
        //   value:
        //     The Boolean value to write (0 or 1).
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(bool value)
        {
        }

        //
        // 摘要:
        //     Writes an unsigned byte to the current stream and advances the stream position
        //     by one byte.
        //
        // 参数:
        //   value:
        //     The unsigned byte to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(byte value)
        {
        }

        //
        // 摘要:
        //     Writes a byte array to the underlying stream.
        //
        // 参数:
        //   buffer:
        //     A byte array containing the data to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        //
        //   T:System.ArgumentNullException:
        //     buffer is null.
        public virtual void Write(byte[] buffer)
        {
        }

        //
        // 摘要:
        //     Writes a region of a byte array to the current stream.
        //
        // 参数:
        //   buffer:
        //     A byte array containing the data to write.
        //
        //   index:
        //     The index of the first byte to read from buffer and to write to the stream.
        //
        //   count:
        //     The number of bytes to read from buffer and to write to the stream.
        //
        // 异常:
        //   T:System.ArgumentException:
        //     The buffer length minus index is less than count.
        //
        //   T:System.ArgumentNullException:
        //     buffer is null.
        //
        //   T:System.ArgumentOutOfRangeException:
        //     index or count is negative.
        //
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(byte[] buffer, int index, int count)
        {
        }

        //
        // 摘要:
        //     Writes a Unicode character to the current stream and advances the current position
        //     of the stream in accordance with the Encoding used and the specific characters
        //     being written to the stream.
        //
        // 参数:
        //   ch:
        //     The non-surrogate, Unicode character to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        //
        //   T:System.ArgumentException:
        //     ch is a single surrogate character.
        public virtual void Write(char ch)
        {
        }

        //
        // 摘要:
        //     Writes a character array to the current stream and advances the current position
        //     of the stream in accordance with the Encoding used and the specific characters
        //     being written to the stream.
        //
        // 参数:
        //   chars:
        //     A character array containing the data to write.
        //
        // 异常:
        //   T:System.ArgumentNullException:
        //     chars is null.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        //
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        public virtual void Write(char[] chars)
        {
        }

        //
        // 摘要:
        //     Writes a section of a character array to the current stream, and advances the
        //     current position of the stream in accordance with the Encoding used and perhaps
        //     the specific characters being written to the stream.
        //
        // 参数:
        //   chars:
        //     A character array containing the data to write.
        //
        //   index:
        //     The index of the first character to read from chars and to write to the stream.
        //
        //   count:
        //     The number of characters to read from chars and to write to the stream.
        //
        // 异常:
        //   T:System.ArgumentException:
        //     The buffer length minus index is less than count.
        //
        //   T:System.ArgumentNullException:
        //     chars is null.
        //
        //   T:System.ArgumentOutOfRangeException:
        //     index or count is negative.
        //
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(char[] chars, int index, int count)
        {
        }

        //
        // 摘要:
        //     Writes a decimal value to the current stream and advances the stream position
        //     by sixteen bytes.
        //
        // 参数:
        //   value:
        //     The decimal value to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(decimal value)
        {
        }

        //
        // 摘要:
        //     Writes an eight-byte floating-point value to the current stream and advances
        //     the stream position by eight bytes.
        //
        // 参数:
        //   value:
        //     The eight-byte floating-point value to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(double value)
        {
        }

        //
        // 摘要:
        //     Writes an two-byte floating-point value to the current stream and advances the
        //     stream position by two bytes.
        //
        // 参数:
        //   value:
        //     The two-byte floating-point value to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(Half value)
        {
        }

        //
        // 摘要:
        //     Writes a two-byte signed integer to the current stream and advances the stream
        //     position by two bytes.
        //
        // 参数:
        //   value:
        //     The two-byte signed integer to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(short value)
        {
        }

        //
        // 摘要:
        //     Writes a four-byte signed integer to the current stream and advances the stream
        //     position by four bytes.
        //
        // 参数:
        //   value:
        //     The four-byte signed integer to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(int value)
        {
        }

        //
        // 摘要:
        //     Writes an eight-byte signed integer to the current stream and advances the stream
        //     position by eight bytes.
        //
        // 参数:
        //   value:
        //     The eight-byte signed integer to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(long value)
        {
        }

        //
        // 摘要:
        //     Writes a span of bytes to the current stream.
        //
        // 参数:
        //   buffer:
        //     The span of bytes to write.
        public virtual void Write(ReadOnlySpan<byte> buffer)
        {
        }

        //
        // 摘要:
        //     Writes a span of characters to the current stream, and advances the current position
        //     of the stream in accordance with the Encoding used and perhaps the specific characters
        //     being written to the stream.
        //
        // 参数:
        //   chars:
        //     A span of chars to write.
        public virtual void Write(ReadOnlySpan<char> chars)
        {
        }

        //
        // 摘要:
        //     Writes a signed byte to the current stream and advances the stream position by
        //     one byte.
        //
        // 参数:
        //   value:
        //     The signed byte to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        [CLSCompliant(false)]
        public virtual void Write(sbyte value)
        {
        }

        //
        // 摘要:
        //     Writes a four-byte floating-point value to the current stream and advances the
        //     stream position by four bytes.
        //
        // 参数:
        //   value:
        //     The four-byte floating-point value to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(float value)
        {
        }

        //
        // 摘要:
        //     Writes a length-prefixed string to this stream in the current encoding of the
        //     System.IO.BinaryWriter, and advances the current position of the stream in accordance
        //     with the encoding used and the specific characters being written to the stream.
        //
        // 参数:
        //   value:
        //     The value to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ArgumentNullException:
        //     value is null.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        public virtual void Write(string value)
        {
        }

        //
        // 摘要:
        //     Writes a two-byte unsigned integer to the current stream and advances the stream
        //     position by two bytes.
        //
        // 参数:
        //   value:
        //     The two-byte unsigned integer to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        [CLSCompliant(false)]
        public virtual void Write(ushort value)
        {
        }

        //
        // 摘要:
        //     Writes a four-byte unsigned integer to the current stream and advances the stream
        //     position by four bytes.
        //
        // 参数:
        //   value:
        //     The four-byte unsigned integer to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        [CLSCompliant(false)]
        public virtual void Write(uint value)
        {
        }

        //
        // 摘要:
        //     Writes an eight-byte unsigned integer to the current stream and advances the
        //     stream position by eight bytes.
        //
        // 参数:
        //   value:
        //     The eight-byte unsigned integer to write.
        //
        // 异常:
        //   T:System.IO.IOException:
        //     An I/O error occurs.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        [CLSCompliant(false)]
        public virtual void Write(ulong value)
        {
        }

        //
        // 摘要:
        //     Writes a 32-bit integer in a compressed format.
        //
        // 参数:
        //   value:
        //     The 32-bit integer to be written.
        //
        // 异常:
        //   T:System.IO.EndOfStreamException:
        //     The end of the stream is reached.
        //
        //   T:System.ObjectDisposedException:
        //     The stream is closed.
        //
        //   T:System.IO.IOException:
        //     The stream is closed.
        public void Write7BitEncodedInt(int value)
        {
        }

        //
        // 摘要:
        //     Writes out a number 7 bits at a time.
        //
        // 参数:
        //   value:
        //     The value to write.
        public void Write7BitEncodedInt64(long value)
        {
        }
    }
}