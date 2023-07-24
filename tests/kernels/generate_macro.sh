. ./functions.sh
. ./kernel_op_config.sh
generated_file=generated_macro.h
ARGS=0
ATTR=0
GenerateHeader
for key in ${!config[@]}; do
    # echo "${config[$key]}"
    ParseOpConfig ${config[$key]}
    ARGS=${op_config[0]}
    ATTR=${op_config[1]}
    GenFlag=${op_config[2]}
    if [ $GenFlag == "GenMacro" ];then
        echo "Generating Macro Utils for $key ..."
        GenerateNncaseTestClassMacro
        GenerateInputMacro
        GenerateGetActualMacro
        GenerateGetExpectMacro
        GenerateNncaseTestBodyMacro
        GenerateNncaseTestSuiteMacro
    fi
done
