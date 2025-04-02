import nncase
import numpy as np
from nncase_base_func import *


def compile_kmodel(model_path, dump_path, calib_data):
    """
    Set compile options and ptq options.
    Compile kmodel.
    Dump the compile-time result to 'compile_options.dump_dir'
    """
    print("\n----------   compile    ----------")
    print("Simplify...")
    model_file = model_simplify(model_path)
    # model_file = model_path

    print("Set options...")
    # import_options
    import_options = nncase.ImportOptions()
    import_options.huggingface_options.use_cache = True
    import_options.huggingface_options.output_attentions = True
    import_options.huggingface_options.output_hidden_states = False

    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = "cpu"  # "cpu"
    compile_options.dump_ir = True  # if False, will not dump the compile-time result.
    compile_options.dump_asm = True
    compile_options.dump_dir = dump_path
    # compile_options.input_file = "/mnt/workspace/onnxruntime-inference-examples/quantization/language_model/llama/smooth_quant/new_model/onnx/llm.onnx.data"
    # compile_options.input_file = "/mnt/model/pipeline/llm-export/model24/onnx/llm.onnx.data"
    
    # TODO: fix this! need change nncase/python/nncase/__init__.py:174L 
    compile_options.cpu_target_options = None
    
    compile_options.shape_bucket_enable = False
    compile_options.shape_bucket_segments_count = 8
    compile_options.shape_bucket_range_info = {}#{"seq_len": [1,512], "history_len": [0,512]}
    compile_options.shape_bucket_fix_var_map = {}#{"seq_len": 9, "history_len": 0}

    # quant
    ptq_options = nncase.PTQTensorOptions()

    ptq_options.quant_type = "uint8"  # datatype : "float32", "int8", "int16"
    ptq_options.w_quant_type = "uint8"  # datatype : "float32", "int8", "int16"
    ptq_options.calibrate_method = "NoClip"  # "Kld"
    ptq_options.finetune_weights_method = "NoFineTuneWeights"
    ptq_options.dump_quant_error = False
    ptq_options.dump_quant_error_symmetric_for_signed = False

    # detail in docs/MixQuant.md
    ptq_options.quant_scheme = ""# "qwen_24_apply_all/try.json"
    ptq_options.export_quant_scheme = False
    ptq_options.export_weight_range_by_channel = False

    ptq_options.samples_count = len(calib_data[0])
    ptq_options.set_tensor_data(calib_data)

    print("Compiling...")
    compiler = nncase.Compiler(compile_options)
    # import
    # model_content = read_model_file(model_file)
    compiler.import_huggingface(model_file, import_options)
    evaluator = compiler.create_evaluator(3)
    
    return evaluator
    # compiler.use_ptq(ptq_options)


    # # compile
    # compiler.compile()
    # kmodel = compiler.gencode_tobytes()
    # print("Write kmodel...")
    # kmodel_path = os.path.join(dump_path, "test.kmodel")
    # with open(kmodel_path, 'wb') as f:
    #     f.write(kmodel)
    # print("----------------end-----------------")
    return kmodel_path

def main(model_path, dump_path, input_file):

    # calib_file = "/media/curio/500_disk/model/issue_model/DF/calib.jpg"
    input_data = [np.fromfile(input_file, dtype=np.int64).reshape(1,20)]

    calib_data = [input_data]

    evaluator = compile_kmodel(model_path, dump_path, calib_data)


    for idx, i in enumerate(input_data):
        evaluator.set_input_tensor(idx, nncase.RuntimeTensor.from_numpy(i))
        print(i.shape)
        i.tofile(os.path.join(dump_path, "input_{}_0.bin".format(idx)))

    evaluator.run()

    for i in range(evaluator.outputs_size):
        result = evaluator.get_output_tensor(i).to_numpy()
        print(result.shape)
        result.tofile(os.path.join(dump_path, "nncase_result_{}.bin".format(i)))

if __name__ == "__main__":
    model_path = "/compiler/yanghaoqi/workspace/nncase/tests/Qwen/Qwen2.5-0.5B-Instruct"
    dump_path = "qwen/"
    input_file = "/compiler/yanghaoqi/workspace/nncase/tests_output/test_qwen2/input/input_0.bin"
    main(model_path, dump_path, input_file)