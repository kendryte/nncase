# new arch k230 set

K230_WORK_PATH=/home/curio/project/k230/rebuild-ir

#${K230_WORK_PATH}/nncase/local_runtime_dir/lib:${K230_WORK_PATH}/k510-gnne-compiler/local_runtime_dir/lib:
export LD_LIBRARY_PATH="${K230_WORK_PATH}/nncase/src/Nncase.Cli/bin/Debug/net7.0/runtimes/linux-x64/native:/usr/local/lib:${LD_LIBRARY_PATH}"
export nncase_DIR="/usr/local/lib/cmake/nncase"
# export NNCASE_CLI=${K230_WORK_PATH}/nncase/src/Nncase.Cli/bin/Debug/net7.0/runtimes/linux-x64/publish/ 
export NNCASE_TARGET_PATH=${K230_WORK_PATH}/k510-gnne-compiler/modules/Nncase.Modules.K230/bin/Debug/net7.0/ 
export NNCASE_COMPILER=${K230_WORK_PATH}/nncase/src/Nncase.Compiler/bin/Debug/net7.0/Nncase.Compiler.dll
export NNCASE_PLUGIN_PATH=${K230_WORK_PATH}/k510-gnne-compiler/modules/Nncase.Modules.K230/bin/Debug/net7.0/

export PYTHONPATH="${K230_WORK_PATH}/nncase/tests:/usr/local/python:/usr/local/lib"
