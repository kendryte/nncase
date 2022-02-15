using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Nncase.Schedule;
namespace Nncase.Simulator
{

    /// <summary>
    /// the kmodel Interpreter. require the nncase runtime lib.
    /// </summary>
    public class Interpreter : IDisposable
    {
        [DllImport("libnncase_csharp")]
        static extern bool interpreter_init();

        [DllImport("libnncase_csharp")]
        static extern unsafe void interpreter_load_model(byte* buffer_ptr, int size);

        [DllImport("libnncase_csharp")]
        static extern nuint interpreter_inputs_size();

        [DllImport("libnncase_csharp")]
        static extern nuint interpreter_outputs_size();

        [DllImport("libnncase_csharp")]
        static extern MemoryRange interpreter_get_input_desc(nuint index);

        [DllImport("libnncase_csharp")]
        static extern MemoryRange interpreter_get_output_desc(nuint index);

        [DllImport("libnncase_csharp")]
        static extern IntPtr interpreter_get_input_tensor(nuint index);

        [DllImport("libnncase_csharp")]
        static extern void interpreter_set_input_tensor(nuint index, IntPtr rt);

        [DllImport("libnncase_csharp")]
        static extern IntPtr interpreter_get_output_tensor(nuint index);

        [DllImport("libnncase_csharp")]
        static extern void interpreter_set_output_tensor(nuint index, IntPtr rt);

        [DllImport("libnncase_csharp")]
        static extern void interpreter_run();

        /// <summary>
        /// ctor
        /// </summary>
        /// <exception cref="InvalidProgramException"></exception>
        public Interpreter()
        {
            if (interpreter_init() == false)
                throw new InvalidProgramException("Only Can Create One Interpreter Instance!");
        }

        /// <summary>
        /// load kmodel from path
        /// </summary>
        /// <param name="model_path"></param>
        public void LoadModel(string model_path)
        {
            LoadModel(File.ReadAllBytes(model_path));
        }

        /// <summary>
        /// load kmodel from content
        /// </summary>
        /// <param name="model_content"></param>
        public unsafe void LoadModel(byte[] model_content)
        {
            fixed (byte* ptr = model_content)
            {
                interpreter_load_model(ptr, model_content.Length);
            }
        }

        /// <summary>
        /// get inputs size
        /// </summary>
        public int InputsSize => (int)interpreter_inputs_size();

        /// <summary>
        /// get output size
        /// </summary>
        public int OuputsSize => (int)interpreter_outputs_size();

        /// <summary>
        /// get input description
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public MemoryRange InputDesc(int index) => interpreter_get_input_desc((nuint)index);

        /// <summary>
        /// get output description
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public MemoryRange OuputDesc(int index) => interpreter_get_output_desc((nuint)index);

        /// <summary>
        /// set the input Tensor
        /// </summary>
        /// <param name="index"></param>
        /// <param name="rt"></param>
        public void InputTensor(int index, RuntimeTensor rt)
        {
            interpreter_set_input_tensor((nuint)index, rt.Handle);
        }

        /// <summary>
        /// set the output Tensor
        /// </summary>
        /// <param name="index"></param>
        /// <param name="rt"></param>
        public void OuputTensor(int index, RuntimeTensor rt)
        {
            interpreter_set_output_tensor((nuint)index, rt.Handle);
        }

        /// <summary>
        /// get the input Tensor
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public RuntimeTensor InputTensor(int index)
        {
            return RuntimeTensor.Create(interpreter_get_input_tensor((nuint)index));
        }


        /// <summary>
        /// get the output Tensor
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public RuntimeTensor OuputTensor(int index)
        {
            return RuntimeTensor.Create(interpreter_get_output_tensor((nuint)index));
        }

        /// <summary>
        /// run the model
        /// </summary>
        public void Run()
        {
            interpreter_run();
        }

        /// <inheritdoc/>
        public void Dispose()
        { }
    }
}