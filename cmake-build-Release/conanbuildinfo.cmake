include(CMakeParseArguments)

macro(conan_find_apple_frameworks FRAMEWORKS_FOUND FRAMEWORKS SUFFIX BUILD_TYPE)
    if(APPLE)
        if(CMAKE_BUILD_TYPE)
            set(_BTYPE ${CMAKE_BUILD_TYPE})
        elseif(NOT BUILD_TYPE STREQUAL "")
            set(_BTYPE ${BUILD_TYPE})
        endif()
        if(_BTYPE)
            if(${_BTYPE} MATCHES "Debug|_DEBUG")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_DEBUG} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_DEBUG} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            elseif(${_BTYPE} MATCHES "Release|_RELEASE")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_RELEASE} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_RELEASE} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            elseif(${_BTYPE} MATCHES "RelWithDebInfo|_RELWITHDEBINFO")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_RELWITHDEBINFO} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_RELWITHDEBINFO} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            elseif(${_BTYPE} MATCHES "MinSizeRel|_MINSIZEREL")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_MINSIZEREL} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_MINSIZEREL} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            endif()
        endif()
        foreach(_FRAMEWORK ${FRAMEWORKS})
            # https://cmake.org/pipermail/cmake-developers/2017-August/030199.html
            find_library(CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND NAMES ${_FRAMEWORK} PATHS ${CONAN_FRAMEWORK_DIRS${SUFFIX}} CMAKE_FIND_ROOT_PATH_BOTH)
            if(CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND)
                list(APPEND ${FRAMEWORKS_FOUND} ${CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND})
            else()
                message(FATAL_ERROR "Framework library ${_FRAMEWORK} not found in paths: ${CONAN_FRAMEWORK_DIRS${SUFFIX}}")
            endif()
        endforeach()
    endif()
endmacro()


#################
###  GSL-LITE
#################
set(CONAN_GSL-LITE_ROOT "/home/curio/.conan/data/gsl-lite/0.37.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_GSL-LITE "/home/curio/.conan/data/gsl-lite/0.37.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(CONAN_LIB_DIRS_GSL-LITE )
set(CONAN_BIN_DIRS_GSL-LITE )
set(CONAN_RES_DIRS_GSL-LITE )
set(CONAN_SRC_DIRS_GSL-LITE )
set(CONAN_BUILD_DIRS_GSL-LITE "/home/curio/.conan/data/gsl-lite/0.37.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/")
set(CONAN_FRAMEWORK_DIRS_GSL-LITE )
set(CONAN_LIBS_GSL-LITE )
set(CONAN_PKG_LIBS_GSL-LITE )
set(CONAN_SYSTEM_LIBS_GSL-LITE )
set(CONAN_FRAMEWORKS_GSL-LITE )
set(CONAN_FRAMEWORKS_FOUND_GSL-LITE "")  # Will be filled later
set(CONAN_DEFINES_GSL-LITE "-DGSL_TERMINATE_ON_CONTRACT_VIOLATION")
set(CONAN_BUILD_MODULES_PATHS_GSL-LITE )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_GSL-LITE "GSL_TERMINATE_ON_CONTRACT_VIOLATION")

set(CONAN_C_FLAGS_GSL-LITE "")
set(CONAN_CXX_FLAGS_GSL-LITE "")
set(CONAN_SHARED_LINKER_FLAGS_GSL-LITE "")
set(CONAN_EXE_LINKER_FLAGS_GSL-LITE "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_GSL-LITE_LIST "")
set(CONAN_CXX_FLAGS_GSL-LITE_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_GSL-LITE_LIST "")
set(CONAN_EXE_LINKER_FLAGS_GSL-LITE_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_GSL-LITE "${CONAN_FRAMEWORKS_GSL-LITE}" "_GSL-LITE" "")
# Append to aggregated values variable
set(CONAN_LIBS_GSL-LITE ${CONAN_PKG_LIBS_GSL-LITE} ${CONAN_SYSTEM_LIBS_GSL-LITE} ${CONAN_FRAMEWORKS_FOUND_GSL-LITE})


#################
###  MPARK-VARIANT
#################
set(CONAN_MPARK-VARIANT_ROOT "/home/curio/.conan/data/mpark-variant/1.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_MPARK-VARIANT "/home/curio/.conan/data/mpark-variant/1.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(CONAN_LIB_DIRS_MPARK-VARIANT )
set(CONAN_BIN_DIRS_MPARK-VARIANT )
set(CONAN_RES_DIRS_MPARK-VARIANT )
set(CONAN_SRC_DIRS_MPARK-VARIANT )
set(CONAN_BUILD_DIRS_MPARK-VARIANT "/home/curio/.conan/data/mpark-variant/1.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/")
set(CONAN_FRAMEWORK_DIRS_MPARK-VARIANT )
set(CONAN_LIBS_MPARK-VARIANT )
set(CONAN_PKG_LIBS_MPARK-VARIANT )
set(CONAN_SYSTEM_LIBS_MPARK-VARIANT )
set(CONAN_FRAMEWORKS_MPARK-VARIANT )
set(CONAN_FRAMEWORKS_FOUND_MPARK-VARIANT "")  # Will be filled later
set(CONAN_DEFINES_MPARK-VARIANT )
set(CONAN_BUILD_MODULES_PATHS_MPARK-VARIANT )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_MPARK-VARIANT )

set(CONAN_C_FLAGS_MPARK-VARIANT "")
set(CONAN_CXX_FLAGS_MPARK-VARIANT "")
set(CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT "")
set(CONAN_EXE_LINKER_FLAGS_MPARK-VARIANT "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_MPARK-VARIANT_LIST "")
set(CONAN_CXX_FLAGS_MPARK-VARIANT_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT_LIST "")
set(CONAN_EXE_LINKER_FLAGS_MPARK-VARIANT_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_MPARK-VARIANT "${CONAN_FRAMEWORKS_MPARK-VARIANT}" "_MPARK-VARIANT" "")
# Append to aggregated values variable
set(CONAN_LIBS_MPARK-VARIANT ${CONAN_PKG_LIBS_MPARK-VARIANT} ${CONAN_SYSTEM_LIBS_MPARK-VARIANT} ${CONAN_FRAMEWORKS_FOUND_MPARK-VARIANT})


#################
###  HKG
#################
set(CONAN_HKG_ROOT "/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_HKG "/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(CONAN_LIB_DIRS_HKG "/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib")
set(CONAN_BIN_DIRS_HKG )
set(CONAN_RES_DIRS_HKG )
set(CONAN_SRC_DIRS_HKG )
set(CONAN_BUILD_DIRS_HKG "/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/")
set(CONAN_FRAMEWORK_DIRS_HKG )
set(CONAN_LIBS_HKG )
set(CONAN_PKG_LIBS_HKG )
set(CONAN_SYSTEM_LIBS_HKG )
set(CONAN_FRAMEWORKS_HKG )
set(CONAN_FRAMEWORKS_FOUND_HKG "")  # Will be filled later
set(CONAN_DEFINES_HKG )
set(CONAN_BUILD_MODULES_PATHS_HKG "/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib/cmake/hkgTargets.cmake"
			"/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib/cmake/hkgHelper.cmake")
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_HKG )

set(CONAN_C_FLAGS_HKG "")
set(CONAN_CXX_FLAGS_HKG "")
set(CONAN_SHARED_LINKER_FLAGS_HKG "")
set(CONAN_EXE_LINKER_FLAGS_HKG "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_HKG_LIST "")
set(CONAN_CXX_FLAGS_HKG_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_HKG_LIST "")
set(CONAN_EXE_LINKER_FLAGS_HKG_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_HKG "${CONAN_FRAMEWORKS_HKG}" "_HKG" "")
# Append to aggregated values variable
set(CONAN_LIBS_HKG ${CONAN_PKG_LIBS_HKG} ${CONAN_SYSTEM_LIBS_HKG} ${CONAN_FRAMEWORKS_FOUND_HKG})


#################
###  PYBIND11
#################
set(CONAN_PYBIND11_ROOT "/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_PYBIND11 "/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/pybind11"
			"/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(CONAN_LIB_DIRS_PYBIND11 "/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib")
set(CONAN_BIN_DIRS_PYBIND11 )
set(CONAN_RES_DIRS_PYBIND11 )
set(CONAN_SRC_DIRS_PYBIND11 )
set(CONAN_BUILD_DIRS_PYBIND11 "/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib/cmake/pybind11")
set(CONAN_FRAMEWORK_DIRS_PYBIND11 )
set(CONAN_LIBS_PYBIND11 )
set(CONAN_PKG_LIBS_PYBIND11 )
set(CONAN_SYSTEM_LIBS_PYBIND11 )
set(CONAN_FRAMEWORKS_PYBIND11 )
set(CONAN_FRAMEWORKS_FOUND_PYBIND11 "")  # Will be filled later
set(CONAN_DEFINES_PYBIND11 )
set(CONAN_BUILD_MODULES_PATHS_PYBIND11 )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_PYBIND11 )

set(CONAN_C_FLAGS_PYBIND11 "")
set(CONAN_CXX_FLAGS_PYBIND11 "")
set(CONAN_SHARED_LINKER_FLAGS_PYBIND11 "")
set(CONAN_EXE_LINKER_FLAGS_PYBIND11 "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_PYBIND11_LIST "")
set(CONAN_CXX_FLAGS_PYBIND11_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_PYBIND11_LIST "")
set(CONAN_EXE_LINKER_FLAGS_PYBIND11_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_PYBIND11 "${CONAN_FRAMEWORKS_PYBIND11}" "_PYBIND11" "")
# Append to aggregated values variable
set(CONAN_LIBS_PYBIND11 ${CONAN_PKG_LIBS_PYBIND11} ${CONAN_SYSTEM_LIBS_PYBIND11} ${CONAN_FRAMEWORKS_FOUND_PYBIND11})


#################
###  ABSEIL
#################
set(CONAN_ABSEIL_ROOT "/home/curio/.conan/data/abseil/20220623.1/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612")
set(CONAN_INCLUDE_DIRS_ABSEIL "/home/curio/.conan/data/abseil/20220623.1/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/include")
set(CONAN_LIB_DIRS_ABSEIL "/home/curio/.conan/data/abseil/20220623.1/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/lib")
set(CONAN_BIN_DIRS_ABSEIL )
set(CONAN_RES_DIRS_ABSEIL )
set(CONAN_SRC_DIRS_ABSEIL )
set(CONAN_BUILD_DIRS_ABSEIL )
set(CONAN_FRAMEWORK_DIRS_ABSEIL )
set(CONAN_LIBS_ABSEIL absl_scoped_set_env absl_failure_signal_handler absl_examine_stack absl_leak_check absl_flags_parse absl_flags_usage absl_flags_usage_internal absl_flags absl_flags_reflection absl_raw_hash_set absl_hashtablez_sampler absl_flags_private_handle_accessor absl_flags_internal absl_flags_config absl_flags_program_name absl_flags_marshalling absl_flags_commandlineflag absl_flags_commandlineflag_internal absl_hash absl_city absl_low_level_hash absl_periodic_sampler absl_random_distributions absl_random_seed_sequences absl_random_internal_pool_urbg absl_random_seed_gen_exception absl_random_internal_seed_material absl_random_internal_randen absl_random_internal_randen_slow absl_random_internal_randen_hwaes absl_random_internal_randen_hwaes_impl absl_random_internal_platform absl_random_internal_distribution_test_util absl_statusor absl_status absl_strerror absl_str_format_internal absl_cordz_sample_token absl_cord absl_cordz_info absl_cord_internal absl_cordz_functions absl_exponential_biased absl_cordz_handle absl_synchronization absl_stacktrace absl_symbolize absl_debugging_internal absl_demangle_internal absl_graphcycles_internal absl_malloc_internal absl_time absl_strings absl_int128 absl_strings_internal absl_base absl_spinlock_wait absl_civil_time absl_time_zone absl_bad_any_cast_impl absl_throw_delegate absl_bad_optional_access absl_bad_variant_access absl_raw_logging_internal absl_log_severity)
set(CONAN_PKG_LIBS_ABSEIL absl_scoped_set_env absl_failure_signal_handler absl_examine_stack absl_leak_check absl_flags_parse absl_flags_usage absl_flags_usage_internal absl_flags absl_flags_reflection absl_raw_hash_set absl_hashtablez_sampler absl_flags_private_handle_accessor absl_flags_internal absl_flags_config absl_flags_program_name absl_flags_marshalling absl_flags_commandlineflag absl_flags_commandlineflag_internal absl_hash absl_city absl_low_level_hash absl_periodic_sampler absl_random_distributions absl_random_seed_sequences absl_random_internal_pool_urbg absl_random_seed_gen_exception absl_random_internal_seed_material absl_random_internal_randen absl_random_internal_randen_slow absl_random_internal_randen_hwaes absl_random_internal_randen_hwaes_impl absl_random_internal_platform absl_random_internal_distribution_test_util absl_statusor absl_status absl_strerror absl_str_format_internal absl_cordz_sample_token absl_cord absl_cordz_info absl_cord_internal absl_cordz_functions absl_exponential_biased absl_cordz_handle absl_synchronization absl_stacktrace absl_symbolize absl_debugging_internal absl_demangle_internal absl_graphcycles_internal absl_malloc_internal absl_time absl_strings absl_int128 absl_strings_internal absl_base absl_spinlock_wait absl_civil_time absl_time_zone absl_bad_any_cast_impl absl_throw_delegate absl_bad_optional_access absl_bad_variant_access absl_raw_logging_internal absl_log_severity)
set(CONAN_SYSTEM_LIBS_ABSEIL pthread m rt)
set(CONAN_FRAMEWORKS_ABSEIL )
set(CONAN_FRAMEWORKS_FOUND_ABSEIL "")  # Will be filled later
set(CONAN_DEFINES_ABSEIL )
set(CONAN_BUILD_MODULES_PATHS_ABSEIL )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_ABSEIL )

set(CONAN_C_FLAGS_ABSEIL "")
set(CONAN_CXX_FLAGS_ABSEIL "")
set(CONAN_SHARED_LINKER_FLAGS_ABSEIL "")
set(CONAN_EXE_LINKER_FLAGS_ABSEIL "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_ABSEIL_LIST "")
set(CONAN_CXX_FLAGS_ABSEIL_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_ABSEIL_LIST "")
set(CONAN_EXE_LINKER_FLAGS_ABSEIL_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_ABSEIL "${CONAN_FRAMEWORKS_ABSEIL}" "_ABSEIL" "")
# Append to aggregated values variable
set(CONAN_LIBS_ABSEIL ${CONAN_PKG_LIBS_ABSEIL} ${CONAN_SYSTEM_LIBS_ABSEIL} ${CONAN_FRAMEWORKS_FOUND_ABSEIL})


#################
###  NETHOST
#################
set(CONAN_NETHOST_ROOT "/home/curio/.conan/data/nethost/6.0.11/_/_/package/4db1be536558d833e52e862fd84d64d75c2b3656")
set(CONAN_INCLUDE_DIRS_NETHOST "/home/curio/.conan/data/nethost/6.0.11/_/_/package/4db1be536558d833e52e862fd84d64d75c2b3656/include")
set(CONAN_LIB_DIRS_NETHOST "/home/curio/.conan/data/nethost/6.0.11/_/_/package/4db1be536558d833e52e862fd84d64d75c2b3656/lib")
set(CONAN_BIN_DIRS_NETHOST )
set(CONAN_RES_DIRS_NETHOST )
set(CONAN_SRC_DIRS_NETHOST )
set(CONAN_BUILD_DIRS_NETHOST "/home/curio/.conan/data/nethost/6.0.11/_/_/package/4db1be536558d833e52e862fd84d64d75c2b3656/")
set(CONAN_FRAMEWORK_DIRS_NETHOST )
set(CONAN_LIBS_NETHOST nethost)
set(CONAN_PKG_LIBS_NETHOST nethost)
set(CONAN_SYSTEM_LIBS_NETHOST )
set(CONAN_FRAMEWORKS_NETHOST )
set(CONAN_FRAMEWORKS_FOUND_NETHOST "")  # Will be filled later
set(CONAN_DEFINES_NETHOST "-DNETHOST_USE_AS_STATIC")
set(CONAN_BUILD_MODULES_PATHS_NETHOST )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_NETHOST "NETHOST_USE_AS_STATIC")

set(CONAN_C_FLAGS_NETHOST "")
set(CONAN_CXX_FLAGS_NETHOST "")
set(CONAN_SHARED_LINKER_FLAGS_NETHOST "")
set(CONAN_EXE_LINKER_FLAGS_NETHOST "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_NETHOST_LIST "")
set(CONAN_CXX_FLAGS_NETHOST_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_NETHOST_LIST "")
set(CONAN_EXE_LINKER_FLAGS_NETHOST_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_NETHOST "${CONAN_FRAMEWORKS_NETHOST}" "_NETHOST" "")
# Append to aggregated values variable
set(CONAN_LIBS_NETHOST ${CONAN_PKG_LIBS_NETHOST} ${CONAN_SYSTEM_LIBS_NETHOST} ${CONAN_FRAMEWORKS_FOUND_NETHOST})


#################
###  MAGIC_ENUM
#################
set(CONAN_MAGIC_ENUM_ROOT "/home/curio/.conan/data/magic_enum/0.7.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_MAGIC_ENUM "/home/curio/.conan/data/magic_enum/0.7.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(CONAN_LIB_DIRS_MAGIC_ENUM )
set(CONAN_BIN_DIRS_MAGIC_ENUM )
set(CONAN_RES_DIRS_MAGIC_ENUM )
set(CONAN_SRC_DIRS_MAGIC_ENUM )
set(CONAN_BUILD_DIRS_MAGIC_ENUM "/home/curio/.conan/data/magic_enum/0.7.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/")
set(CONAN_FRAMEWORK_DIRS_MAGIC_ENUM )
set(CONAN_LIBS_MAGIC_ENUM )
set(CONAN_PKG_LIBS_MAGIC_ENUM )
set(CONAN_SYSTEM_LIBS_MAGIC_ENUM )
set(CONAN_FRAMEWORKS_MAGIC_ENUM )
set(CONAN_FRAMEWORKS_FOUND_MAGIC_ENUM "")  # Will be filled later
set(CONAN_DEFINES_MAGIC_ENUM )
set(CONAN_BUILD_MODULES_PATHS_MAGIC_ENUM )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_MAGIC_ENUM )

set(CONAN_C_FLAGS_MAGIC_ENUM "")
set(CONAN_CXX_FLAGS_MAGIC_ENUM "")
set(CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM "")
set(CONAN_EXE_LINKER_FLAGS_MAGIC_ENUM "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_MAGIC_ENUM_LIST "")
set(CONAN_CXX_FLAGS_MAGIC_ENUM_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM_LIST "")
set(CONAN_EXE_LINKER_FLAGS_MAGIC_ENUM_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_MAGIC_ENUM "${CONAN_FRAMEWORKS_MAGIC_ENUM}" "_MAGIC_ENUM" "")
# Append to aggregated values variable
set(CONAN_LIBS_MAGIC_ENUM ${CONAN_PKG_LIBS_MAGIC_ENUM} ${CONAN_SYSTEM_LIBS_MAGIC_ENUM} ${CONAN_FRAMEWORKS_FOUND_MAGIC_ENUM})


#################
###  SPDLOG
#################
set(CONAN_SPDLOG_ROOT "/home/curio/.conan/data/spdlog/1.8.2/_/_/package/2f8d6866984cf9c9262a45a6675ee1fab9a81fe2")
set(CONAN_INCLUDE_DIRS_SPDLOG "/home/curio/.conan/data/spdlog/1.8.2/_/_/package/2f8d6866984cf9c9262a45a6675ee1fab9a81fe2/include")
set(CONAN_LIB_DIRS_SPDLOG "/home/curio/.conan/data/spdlog/1.8.2/_/_/package/2f8d6866984cf9c9262a45a6675ee1fab9a81fe2/lib")
set(CONAN_BIN_DIRS_SPDLOG )
set(CONAN_RES_DIRS_SPDLOG )
set(CONAN_SRC_DIRS_SPDLOG )
set(CONAN_BUILD_DIRS_SPDLOG "/home/curio/.conan/data/spdlog/1.8.2/_/_/package/2f8d6866984cf9c9262a45a6675ee1fab9a81fe2/")
set(CONAN_FRAMEWORK_DIRS_SPDLOG )
set(CONAN_LIBS_SPDLOG spdlog)
set(CONAN_PKG_LIBS_SPDLOG spdlog)
set(CONAN_SYSTEM_LIBS_SPDLOG pthread)
set(CONAN_FRAMEWORKS_SPDLOG )
set(CONAN_FRAMEWORKS_FOUND_SPDLOG "")  # Will be filled later
set(CONAN_DEFINES_SPDLOG "-DSPDLOG_COMPILED_LIB"
			"-DSPDLOG_FMT_EXTERNAL")
set(CONAN_BUILD_MODULES_PATHS_SPDLOG )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_SPDLOG "SPDLOG_COMPILED_LIB"
			"SPDLOG_FMT_EXTERNAL")

set(CONAN_C_FLAGS_SPDLOG "")
set(CONAN_CXX_FLAGS_SPDLOG "")
set(CONAN_SHARED_LINKER_FLAGS_SPDLOG "")
set(CONAN_EXE_LINKER_FLAGS_SPDLOG "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_SPDLOG_LIST "")
set(CONAN_CXX_FLAGS_SPDLOG_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_SPDLOG_LIST "")
set(CONAN_EXE_LINKER_FLAGS_SPDLOG_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_SPDLOG "${CONAN_FRAMEWORKS_SPDLOG}" "_SPDLOG" "")
# Append to aggregated values variable
set(CONAN_LIBS_SPDLOG ${CONAN_PKG_LIBS_SPDLOG} ${CONAN_SYSTEM_LIBS_SPDLOG} ${CONAN_FRAMEWORKS_FOUND_SPDLOG})


#################
###  INJA
#################
set(CONAN_INJA_ROOT "/home/curio/.conan/data/inja/3.2.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_INJA "/home/curio/.conan/data/inja/3.2.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(CONAN_LIB_DIRS_INJA )
set(CONAN_BIN_DIRS_INJA )
set(CONAN_RES_DIRS_INJA )
set(CONAN_SRC_DIRS_INJA )
set(CONAN_BUILD_DIRS_INJA )
set(CONAN_FRAMEWORK_DIRS_INJA )
set(CONAN_LIBS_INJA )
set(CONAN_PKG_LIBS_INJA )
set(CONAN_SYSTEM_LIBS_INJA )
set(CONAN_FRAMEWORKS_INJA )
set(CONAN_FRAMEWORKS_FOUND_INJA "")  # Will be filled later
set(CONAN_DEFINES_INJA )
set(CONAN_BUILD_MODULES_PATHS_INJA )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_INJA )

set(CONAN_C_FLAGS_INJA "")
set(CONAN_CXX_FLAGS_INJA "")
set(CONAN_SHARED_LINKER_FLAGS_INJA "")
set(CONAN_EXE_LINKER_FLAGS_INJA "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_INJA_LIST "")
set(CONAN_CXX_FLAGS_INJA_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_INJA_LIST "")
set(CONAN_EXE_LINKER_FLAGS_INJA_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_INJA "${CONAN_FRAMEWORKS_INJA}" "_INJA" "")
# Append to aggregated values variable
set(CONAN_LIBS_INJA ${CONAN_PKG_LIBS_INJA} ${CONAN_SYSTEM_LIBS_INJA} ${CONAN_FRAMEWORKS_FOUND_INJA})


#################
###  VULKAN-LOADER
#################
set(CONAN_VULKAN-LOADER_ROOT "/home/curio/.conan/data/vulkan-loader/1.2.182/_/_/package/56ab386b83e5f1276a7374d6a809c723c88dc6aa")
set(CONAN_INCLUDE_DIRS_VULKAN-LOADER "/home/curio/.conan/data/vulkan-headers/1.2.182/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/curio/.conan/data/vulkan-headers/1.2.182/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/res/vulkan/registry")
set(CONAN_LIB_DIRS_VULKAN-LOADER "/home/curio/.conan/data/vulkan-loader/1.2.182/_/_/package/56ab386b83e5f1276a7374d6a809c723c88dc6aa/lib")
set(CONAN_BIN_DIRS_VULKAN-LOADER )
set(CONAN_RES_DIRS_VULKAN-LOADER )
set(CONAN_SRC_DIRS_VULKAN-LOADER )
set(CONAN_BUILD_DIRS_VULKAN-LOADER "/home/curio/.conan/data/vulkan-loader/1.2.182/_/_/package/56ab386b83e5f1276a7374d6a809c723c88dc6aa/")
set(CONAN_FRAMEWORK_DIRS_VULKAN-LOADER )
set(CONAN_LIBS_VULKAN-LOADER vulkan)
set(CONAN_PKG_LIBS_VULKAN-LOADER vulkan)
set(CONAN_SYSTEM_LIBS_VULKAN-LOADER dl pthread m)
set(CONAN_FRAMEWORKS_VULKAN-LOADER )
set(CONAN_FRAMEWORKS_FOUND_VULKAN-LOADER "")  # Will be filled later
set(CONAN_DEFINES_VULKAN-LOADER )
set(CONAN_BUILD_MODULES_PATHS_VULKAN-LOADER )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_VULKAN-LOADER )

set(CONAN_C_FLAGS_VULKAN-LOADER "")
set(CONAN_CXX_FLAGS_VULKAN-LOADER "")
set(CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER "")
set(CONAN_EXE_LINKER_FLAGS_VULKAN-LOADER "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_VULKAN-LOADER_LIST "")
set(CONAN_CXX_FLAGS_VULKAN-LOADER_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER_LIST "")
set(CONAN_EXE_LINKER_FLAGS_VULKAN-LOADER_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_VULKAN-LOADER "${CONAN_FRAMEWORKS_VULKAN-LOADER}" "_VULKAN-LOADER" "")
# Append to aggregated values variable
set(CONAN_LIBS_VULKAN-LOADER ${CONAN_PKG_LIBS_VULKAN-LOADER} ${CONAN_SYSTEM_LIBS_VULKAN-LOADER} ${CONAN_FRAMEWORKS_FOUND_VULKAN-LOADER})


#################
###  FMT
#################
set(CONAN_FMT_ROOT "/home/curio/.conan/data/fmt/7.1.3/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612")
set(CONAN_INCLUDE_DIRS_FMT "/home/curio/.conan/data/fmt/7.1.3/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/include")
set(CONAN_LIB_DIRS_FMT "/home/curio/.conan/data/fmt/7.1.3/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/lib")
set(CONAN_BIN_DIRS_FMT )
set(CONAN_RES_DIRS_FMT )
set(CONAN_SRC_DIRS_FMT )
set(CONAN_BUILD_DIRS_FMT "/home/curio/.conan/data/fmt/7.1.3/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/")
set(CONAN_FRAMEWORK_DIRS_FMT )
set(CONAN_LIBS_FMT fmt)
set(CONAN_PKG_LIBS_FMT fmt)
set(CONAN_SYSTEM_LIBS_FMT )
set(CONAN_FRAMEWORKS_FMT )
set(CONAN_FRAMEWORKS_FOUND_FMT "")  # Will be filled later
set(CONAN_DEFINES_FMT )
set(CONAN_BUILD_MODULES_PATHS_FMT )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_FMT )

set(CONAN_C_FLAGS_FMT "")
set(CONAN_CXX_FLAGS_FMT "")
set(CONAN_SHARED_LINKER_FLAGS_FMT "")
set(CONAN_EXE_LINKER_FLAGS_FMT "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_FMT_LIST "")
set(CONAN_CXX_FLAGS_FMT_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_FMT_LIST "")
set(CONAN_EXE_LINKER_FLAGS_FMT_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_FMT "${CONAN_FRAMEWORKS_FMT}" "_FMT" "")
# Append to aggregated values variable
set(CONAN_LIBS_FMT ${CONAN_PKG_LIBS_FMT} ${CONAN_SYSTEM_LIBS_FMT} ${CONAN_FRAMEWORKS_FOUND_FMT})


#################
###  NLOHMANN_JSON
#################
set(CONAN_NLOHMANN_JSON_ROOT "/home/curio/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_NLOHMANN_JSON "/home/curio/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(CONAN_LIB_DIRS_NLOHMANN_JSON )
set(CONAN_BIN_DIRS_NLOHMANN_JSON )
set(CONAN_RES_DIRS_NLOHMANN_JSON )
set(CONAN_SRC_DIRS_NLOHMANN_JSON )
set(CONAN_BUILD_DIRS_NLOHMANN_JSON "/home/curio/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/")
set(CONAN_FRAMEWORK_DIRS_NLOHMANN_JSON )
set(CONAN_LIBS_NLOHMANN_JSON )
set(CONAN_PKG_LIBS_NLOHMANN_JSON )
set(CONAN_SYSTEM_LIBS_NLOHMANN_JSON )
set(CONAN_FRAMEWORKS_NLOHMANN_JSON )
set(CONAN_FRAMEWORKS_FOUND_NLOHMANN_JSON "")  # Will be filled later
set(CONAN_DEFINES_NLOHMANN_JSON )
set(CONAN_BUILD_MODULES_PATHS_NLOHMANN_JSON )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_NLOHMANN_JSON )

set(CONAN_C_FLAGS_NLOHMANN_JSON "")
set(CONAN_CXX_FLAGS_NLOHMANN_JSON "")
set(CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON "")
set(CONAN_EXE_LINKER_FLAGS_NLOHMANN_JSON "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_NLOHMANN_JSON_LIST "")
set(CONAN_CXX_FLAGS_NLOHMANN_JSON_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON_LIST "")
set(CONAN_EXE_LINKER_FLAGS_NLOHMANN_JSON_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_NLOHMANN_JSON "${CONAN_FRAMEWORKS_NLOHMANN_JSON}" "_NLOHMANN_JSON" "")
# Append to aggregated values variable
set(CONAN_LIBS_NLOHMANN_JSON ${CONAN_PKG_LIBS_NLOHMANN_JSON} ${CONAN_SYSTEM_LIBS_NLOHMANN_JSON} ${CONAN_FRAMEWORKS_FOUND_NLOHMANN_JSON})


#################
###  VULKAN-HEADERS
#################
set(CONAN_VULKAN-HEADERS_ROOT "/home/curio/.conan/data/vulkan-headers/1.2.182/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_VULKAN-HEADERS "/home/curio/.conan/data/vulkan-headers/1.2.182/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/curio/.conan/data/vulkan-headers/1.2.182/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/res/vulkan/registry")
set(CONAN_LIB_DIRS_VULKAN-HEADERS )
set(CONAN_BIN_DIRS_VULKAN-HEADERS )
set(CONAN_RES_DIRS_VULKAN-HEADERS "/home/curio/.conan/data/vulkan-headers/1.2.182/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/res")
set(CONAN_SRC_DIRS_VULKAN-HEADERS )
set(CONAN_BUILD_DIRS_VULKAN-HEADERS "/home/curio/.conan/data/vulkan-headers/1.2.182/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/")
set(CONAN_FRAMEWORK_DIRS_VULKAN-HEADERS )
set(CONAN_LIBS_VULKAN-HEADERS )
set(CONAN_PKG_LIBS_VULKAN-HEADERS )
set(CONAN_SYSTEM_LIBS_VULKAN-HEADERS )
set(CONAN_FRAMEWORKS_VULKAN-HEADERS )
set(CONAN_FRAMEWORKS_FOUND_VULKAN-HEADERS "")  # Will be filled later
set(CONAN_DEFINES_VULKAN-HEADERS )
set(CONAN_BUILD_MODULES_PATHS_VULKAN-HEADERS )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_VULKAN-HEADERS )

set(CONAN_C_FLAGS_VULKAN-HEADERS "")
set(CONAN_CXX_FLAGS_VULKAN-HEADERS "")
set(CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS "")
set(CONAN_EXE_LINKER_FLAGS_VULKAN-HEADERS "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_VULKAN-HEADERS_LIST "")
set(CONAN_CXX_FLAGS_VULKAN-HEADERS_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS_LIST "")
set(CONAN_EXE_LINKER_FLAGS_VULKAN-HEADERS_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_VULKAN-HEADERS "${CONAN_FRAMEWORKS_VULKAN-HEADERS}" "_VULKAN-HEADERS" "")
# Append to aggregated values variable
set(CONAN_LIBS_VULKAN-HEADERS ${CONAN_PKG_LIBS_VULKAN-HEADERS} ${CONAN_SYSTEM_LIBS_VULKAN-HEADERS} ${CONAN_FRAMEWORKS_FOUND_VULKAN-HEADERS})


### Definition of global aggregated variables ###

set(CONAN_PACKAGE_NAME None)
set(CONAN_PACKAGE_VERSION None)

set(CONAN_SETTINGS_ARCH "x86_64")
set(CONAN_SETTINGS_BUILD_TYPE "Release")
set(CONAN_SETTINGS_COMPILER "gcc")
set(CONAN_SETTINGS_COMPILER_CPPSTD "20")
set(CONAN_SETTINGS_COMPILER_LIBCXX "libstdc++11")
set(CONAN_SETTINGS_COMPILER_VERSION "10")
set(CONAN_SETTINGS_OS "Linux")

set(CONAN_DEPENDENCIES gsl-lite mpark-variant hkg pybind11 abseil nethost magic_enum spdlog inja vulkan-loader fmt nlohmann_json vulkan-headers)
# Storing original command line args (CMake helper) flags
set(CONAN_CMD_CXX_FLAGS ${CONAN_CXX_FLAGS})

set(CONAN_CMD_SHARED_LINKER_FLAGS ${CONAN_SHARED_LINKER_FLAGS})
set(CONAN_CMD_C_FLAGS ${CONAN_C_FLAGS})
# Defining accumulated conan variables for all deps

set(CONAN_INCLUDE_DIRS "/home/curio/.conan/data/gsl-lite/0.37.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/curio/.conan/data/mpark-variant/1.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/pybind11"
			"/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/curio/.conan/data/abseil/20220623.1/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/include"
			"/home/curio/.conan/data/nethost/6.0.11/_/_/package/4db1be536558d833e52e862fd84d64d75c2b3656/include"
			"/home/curio/.conan/data/magic_enum/0.7.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/curio/.conan/data/spdlog/1.8.2/_/_/package/2f8d6866984cf9c9262a45a6675ee1fab9a81fe2/include"
			"/home/curio/.conan/data/inja/3.2.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/curio/.conan/data/fmt/7.1.3/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/include"
			"/home/curio/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/curio/.conan/data/vulkan-headers/1.2.182/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/curio/.conan/data/vulkan-headers/1.2.182/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/res/vulkan/registry" ${CONAN_INCLUDE_DIRS})
set(CONAN_LIB_DIRS "/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib"
			"/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib"
			"/home/curio/.conan/data/abseil/20220623.1/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/lib"
			"/home/curio/.conan/data/nethost/6.0.11/_/_/package/4db1be536558d833e52e862fd84d64d75c2b3656/lib"
			"/home/curio/.conan/data/spdlog/1.8.2/_/_/package/2f8d6866984cf9c9262a45a6675ee1fab9a81fe2/lib"
			"/home/curio/.conan/data/vulkan-loader/1.2.182/_/_/package/56ab386b83e5f1276a7374d6a809c723c88dc6aa/lib"
			"/home/curio/.conan/data/fmt/7.1.3/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/lib" ${CONAN_LIB_DIRS})
set(CONAN_BIN_DIRS  ${CONAN_BIN_DIRS})
set(CONAN_RES_DIRS "/home/curio/.conan/data/vulkan-headers/1.2.182/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/res" ${CONAN_RES_DIRS})
set(CONAN_FRAMEWORK_DIRS  ${CONAN_FRAMEWORK_DIRS})
set(CONAN_LIBS absl_scoped_set_env absl_failure_signal_handler absl_examine_stack absl_leak_check absl_flags_parse absl_flags_usage absl_flags_usage_internal absl_flags absl_flags_reflection absl_raw_hash_set absl_hashtablez_sampler absl_flags_private_handle_accessor absl_flags_internal absl_flags_config absl_flags_program_name absl_flags_marshalling absl_flags_commandlineflag absl_flags_commandlineflag_internal absl_hash absl_city absl_low_level_hash absl_periodic_sampler absl_random_distributions absl_random_seed_sequences absl_random_internal_pool_urbg absl_random_seed_gen_exception absl_random_internal_seed_material absl_random_internal_randen absl_random_internal_randen_slow absl_random_internal_randen_hwaes absl_random_internal_randen_hwaes_impl absl_random_internal_platform absl_random_internal_distribution_test_util absl_statusor absl_status absl_strerror absl_str_format_internal absl_cordz_sample_token absl_cord absl_cordz_info absl_cord_internal absl_cordz_functions absl_exponential_biased absl_cordz_handle absl_synchronization absl_stacktrace absl_symbolize absl_debugging_internal absl_demangle_internal absl_graphcycles_internal absl_malloc_internal absl_time absl_strings absl_int128 absl_strings_internal absl_base absl_spinlock_wait absl_civil_time absl_time_zone absl_bad_any_cast_impl absl_throw_delegate absl_bad_optional_access absl_bad_variant_access absl_raw_logging_internal absl_log_severity nethost spdlog vulkan fmt ${CONAN_LIBS})
set(CONAN_PKG_LIBS absl_scoped_set_env absl_failure_signal_handler absl_examine_stack absl_leak_check absl_flags_parse absl_flags_usage absl_flags_usage_internal absl_flags absl_flags_reflection absl_raw_hash_set absl_hashtablez_sampler absl_flags_private_handle_accessor absl_flags_internal absl_flags_config absl_flags_program_name absl_flags_marshalling absl_flags_commandlineflag absl_flags_commandlineflag_internal absl_hash absl_city absl_low_level_hash absl_periodic_sampler absl_random_distributions absl_random_seed_sequences absl_random_internal_pool_urbg absl_random_seed_gen_exception absl_random_internal_seed_material absl_random_internal_randen absl_random_internal_randen_slow absl_random_internal_randen_hwaes absl_random_internal_randen_hwaes_impl absl_random_internal_platform absl_random_internal_distribution_test_util absl_statusor absl_status absl_strerror absl_str_format_internal absl_cordz_sample_token absl_cord absl_cordz_info absl_cord_internal absl_cordz_functions absl_exponential_biased absl_cordz_handle absl_synchronization absl_stacktrace absl_symbolize absl_debugging_internal absl_demangle_internal absl_graphcycles_internal absl_malloc_internal absl_time absl_strings absl_int128 absl_strings_internal absl_base absl_spinlock_wait absl_civil_time absl_time_zone absl_bad_any_cast_impl absl_throw_delegate absl_bad_optional_access absl_bad_variant_access absl_raw_logging_internal absl_log_severity nethost spdlog vulkan fmt ${CONAN_PKG_LIBS})
set(CONAN_SYSTEM_LIBS rt dl pthread m ${CONAN_SYSTEM_LIBS})
set(CONAN_FRAMEWORKS  ${CONAN_FRAMEWORKS})
set(CONAN_FRAMEWORKS_FOUND "")  # Will be filled later
set(CONAN_DEFINES "-DSPDLOG_COMPILED_LIB"
			"-DSPDLOG_FMT_EXTERNAL"
			"-DNETHOST_USE_AS_STATIC"
			"-DGSL_TERMINATE_ON_CONTRACT_VIOLATION" ${CONAN_DEFINES})
set(CONAN_BUILD_MODULES_PATHS "/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib/cmake/hkgTargets.cmake"
			"/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib/cmake/hkgHelper.cmake" ${CONAN_BUILD_MODULES_PATHS})
set(CONAN_CMAKE_MODULE_PATH "/home/curio/.conan/data/gsl-lite/0.37.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/curio/.conan/data/mpark-variant/1.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib/cmake/pybind11"
			"/home/curio/.conan/data/nethost/6.0.11/_/_/package/4db1be536558d833e52e862fd84d64d75c2b3656/"
			"/home/curio/.conan/data/magic_enum/0.7.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/curio/.conan/data/spdlog/1.8.2/_/_/package/2f8d6866984cf9c9262a45a6675ee1fab9a81fe2/"
			"/home/curio/.conan/data/vulkan-loader/1.2.182/_/_/package/56ab386b83e5f1276a7374d6a809c723c88dc6aa/"
			"/home/curio/.conan/data/fmt/7.1.3/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/"
			"/home/curio/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/curio/.conan/data/vulkan-headers/1.2.182/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/" ${CONAN_CMAKE_MODULE_PATH})

set(CONAN_CXX_FLAGS " ${CONAN_CXX_FLAGS}")
set(CONAN_SHARED_LINKER_FLAGS " ${CONAN_SHARED_LINKER_FLAGS}")
set(CONAN_EXE_LINKER_FLAGS " ${CONAN_EXE_LINKER_FLAGS}")
set(CONAN_C_FLAGS " ${CONAN_C_FLAGS}")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND "${CONAN_FRAMEWORKS}" "" "")
# Append to aggregated values variable: Use CONAN_LIBS instead of CONAN_PKG_LIBS to include user appended vars
set(CONAN_LIBS ${CONAN_LIBS} ${CONAN_SYSTEM_LIBS} ${CONAN_FRAMEWORKS_FOUND})


###  Definition of macros and functions ###

macro(conan_define_targets)
    if(${CMAKE_VERSION} VERSION_LESS "3.1.2")
        message(FATAL_ERROR "TARGETS not supported by your CMake version!")
    endif()  # CMAKE > 3.x
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CONAN_CMD_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CONAN_CMD_C_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${CONAN_CMD_SHARED_LINKER_FLAGS}")


    set(_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES "${CONAN_SYSTEM_LIBS_GSL-LITE} ${CONAN_FRAMEWORKS_FOUND_GSL-LITE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES "${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_GSL-LITE}" "${CONAN_LIB_DIRS_GSL-LITE}"
                                  CONAN_PACKAGE_TARGETS_GSL-LITE "${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES}"
                                  "" gsl-lite)
    set(_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_GSL-LITE_DEBUG} ${CONAN_FRAMEWORKS_FOUND_GSL-LITE_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_GSL-LITE_DEBUG}" "${CONAN_LIB_DIRS_GSL-LITE_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_GSL-LITE_DEBUG "${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_DEBUG}"
                                  "debug" gsl-lite)
    set(_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_GSL-LITE_RELEASE} ${CONAN_FRAMEWORKS_FOUND_GSL-LITE_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_GSL-LITE_RELEASE}" "${CONAN_LIB_DIRS_GSL-LITE_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_GSL-LITE_RELEASE "${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_RELEASE}"
                                  "release" gsl-lite)
    set(_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_GSL-LITE_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_GSL-LITE_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_GSL-LITE_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_GSL-LITE_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_GSL-LITE_RELWITHDEBINFO "${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" gsl-lite)
    set(_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_GSL-LITE_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_GSL-LITE_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_GSL-LITE_MINSIZEREL}" "${CONAN_LIB_DIRS_GSL-LITE_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_GSL-LITE_MINSIZEREL "${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" gsl-lite)

    add_library(CONAN_PKG::gsl-lite INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::gsl-lite PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_GSL-LITE} ${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GSL-LITE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GSL-LITE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_GSL-LITE_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_GSL-LITE_RELEASE} ${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GSL-LITE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GSL-LITE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_GSL-LITE_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_GSL-LITE_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GSL-LITE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GSL-LITE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_GSL-LITE_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_GSL-LITE_MINSIZEREL} ${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GSL-LITE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GSL-LITE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_GSL-LITE_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_GSL-LITE_DEBUG} ${_CONAN_PKG_LIBS_GSL-LITE_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GSL-LITE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GSL-LITE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_GSL-LITE_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::gsl-lite PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_GSL-LITE}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_GSL-LITE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_GSL-LITE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_GSL-LITE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_GSL-LITE_DEBUG}>)
    set_property(TARGET CONAN_PKG::gsl-lite PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_GSL-LITE}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_GSL-LITE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_GSL-LITE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_GSL-LITE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_GSL-LITE_DEBUG}>)
    set_property(TARGET CONAN_PKG::gsl-lite PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_GSL-LITE_LIST} ${CONAN_CXX_FLAGS_GSL-LITE_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_GSL-LITE_RELEASE_LIST} ${CONAN_CXX_FLAGS_GSL-LITE_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_GSL-LITE_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_GSL-LITE_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_GSL-LITE_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_GSL-LITE_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_GSL-LITE_DEBUG_LIST}  ${CONAN_CXX_FLAGS_GSL-LITE_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES "${CONAN_SYSTEM_LIBS_MPARK-VARIANT} ${CONAN_FRAMEWORKS_FOUND_MPARK-VARIANT} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES "${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_MPARK-VARIANT}" "${CONAN_LIB_DIRS_MPARK-VARIANT}"
                                  CONAN_PACKAGE_TARGETS_MPARK-VARIANT "${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES}"
                                  "" mpark-variant)
    set(_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_MPARK-VARIANT_DEBUG} ${CONAN_FRAMEWORKS_FOUND_MPARK-VARIANT_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_MPARK-VARIANT_DEBUG}" "${CONAN_LIB_DIRS_MPARK-VARIANT_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_MPARK-VARIANT_DEBUG "${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_DEBUG}"
                                  "debug" mpark-variant)
    set(_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_MPARK-VARIANT_RELEASE} ${CONAN_FRAMEWORKS_FOUND_MPARK-VARIANT_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_MPARK-VARIANT_RELEASE}" "${CONAN_LIB_DIRS_MPARK-VARIANT_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_MPARK-VARIANT_RELEASE "${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_RELEASE}"
                                  "release" mpark-variant)
    set(_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_MPARK-VARIANT_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_MPARK-VARIANT_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_MPARK-VARIANT_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_MPARK-VARIANT_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_MPARK-VARIANT_RELWITHDEBINFO "${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" mpark-variant)
    set(_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_MPARK-VARIANT_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_MPARK-VARIANT_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_MPARK-VARIANT_MINSIZEREL}" "${CONAN_LIB_DIRS_MPARK-VARIANT_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_MPARK-VARIANT_MINSIZEREL "${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" mpark-variant)

    add_library(CONAN_PKG::mpark-variant INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::mpark-variant PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_MPARK-VARIANT} ${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_MPARK-VARIANT_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_MPARK-VARIANT_RELEASE} ${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_MPARK-VARIANT_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_MPARK-VARIANT_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_MPARK-VARIANT_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_MPARK-VARIANT_MINSIZEREL} ${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_MPARK-VARIANT_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_MPARK-VARIANT_DEBUG} ${_CONAN_PKG_LIBS_MPARK-VARIANT_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MPARK-VARIANT_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_MPARK-VARIANT_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::mpark-variant PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_MPARK-VARIANT}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_MPARK-VARIANT_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_MPARK-VARIANT_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_MPARK-VARIANT_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_MPARK-VARIANT_DEBUG}>)
    set_property(TARGET CONAN_PKG::mpark-variant PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_MPARK-VARIANT}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_MPARK-VARIANT_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_MPARK-VARIANT_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_MPARK-VARIANT_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_MPARK-VARIANT_DEBUG}>)
    set_property(TARGET CONAN_PKG::mpark-variant PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_MPARK-VARIANT_LIST} ${CONAN_CXX_FLAGS_MPARK-VARIANT_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_MPARK-VARIANT_RELEASE_LIST} ${CONAN_CXX_FLAGS_MPARK-VARIANT_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_MPARK-VARIANT_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_MPARK-VARIANT_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_MPARK-VARIANT_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_MPARK-VARIANT_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_MPARK-VARIANT_DEBUG_LIST}  ${CONAN_CXX_FLAGS_MPARK-VARIANT_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_HKG_DEPENDENCIES "${CONAN_SYSTEM_LIBS_HKG} ${CONAN_FRAMEWORKS_FOUND_HKG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_HKG_DEPENDENCIES "${_CONAN_PKG_LIBS_HKG_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_HKG}" "${CONAN_LIB_DIRS_HKG}"
                                  CONAN_PACKAGE_TARGETS_HKG "${_CONAN_PKG_LIBS_HKG_DEPENDENCIES}"
                                  "" hkg)
    set(_CONAN_PKG_LIBS_HKG_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_HKG_DEBUG} ${CONAN_FRAMEWORKS_FOUND_HKG_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_HKG_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_HKG_DEBUG}" "${CONAN_LIB_DIRS_HKG_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_HKG_DEBUG "${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_DEBUG}"
                                  "debug" hkg)
    set(_CONAN_PKG_LIBS_HKG_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_HKG_RELEASE} ${CONAN_FRAMEWORKS_FOUND_HKG_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_HKG_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_HKG_RELEASE}" "${CONAN_LIB_DIRS_HKG_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_HKG_RELEASE "${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_RELEASE}"
                                  "release" hkg)
    set(_CONAN_PKG_LIBS_HKG_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_HKG_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_HKG_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_HKG_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_HKG_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_HKG_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_HKG_RELWITHDEBINFO "${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" hkg)
    set(_CONAN_PKG_LIBS_HKG_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_HKG_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_HKG_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_HKG_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_HKG_MINSIZEREL}" "${CONAN_LIB_DIRS_HKG_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_HKG_MINSIZEREL "${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" hkg)

    add_library(CONAN_PKG::hkg INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::hkg PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_HKG} ${_CONAN_PKG_LIBS_HKG_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_HKG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_HKG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_HKG_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_HKG_RELEASE} ${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_HKG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_HKG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_HKG_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_HKG_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_HKG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_HKG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_HKG_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_HKG_MINSIZEREL} ${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_HKG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_HKG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_HKG_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_HKG_DEBUG} ${_CONAN_PKG_LIBS_HKG_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_HKG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_HKG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_HKG_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::hkg PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_HKG}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_HKG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_HKG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_HKG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_HKG_DEBUG}>)
    set_property(TARGET CONAN_PKG::hkg PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_HKG}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_HKG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_HKG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_HKG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_HKG_DEBUG}>)
    set_property(TARGET CONAN_PKG::hkg PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_HKG_LIST} ${CONAN_CXX_FLAGS_HKG_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_HKG_RELEASE_LIST} ${CONAN_CXX_FLAGS_HKG_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_HKG_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_HKG_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_HKG_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_HKG_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_HKG_DEBUG_LIST}  ${CONAN_CXX_FLAGS_HKG_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES "${CONAN_SYSTEM_LIBS_PYBIND11} ${CONAN_FRAMEWORKS_FOUND_PYBIND11} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PYBIND11}" "${CONAN_LIB_DIRS_PYBIND11}"
                                  CONAN_PACKAGE_TARGETS_PYBIND11 "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES}"
                                  "" pybind11)
    set(_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_PYBIND11_DEBUG} ${CONAN_FRAMEWORKS_FOUND_PYBIND11_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PYBIND11_DEBUG}" "${CONAN_LIB_DIRS_PYBIND11_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_PYBIND11_DEBUG "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_DEBUG}"
                                  "debug" pybind11)
    set(_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_PYBIND11_RELEASE} ${CONAN_FRAMEWORKS_FOUND_PYBIND11_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PYBIND11_RELEASE}" "${CONAN_LIB_DIRS_PYBIND11_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_PYBIND11_RELEASE "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELEASE}"
                                  "release" pybind11)
    set(_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_PYBIND11_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_PYBIND11_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PYBIND11_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_PYBIND11_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_PYBIND11_RELWITHDEBINFO "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" pybind11)
    set(_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_PYBIND11_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_PYBIND11_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PYBIND11_MINSIZEREL}" "${CONAN_LIB_DIRS_PYBIND11_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_PYBIND11_MINSIZEREL "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" pybind11)

    add_library(CONAN_PKG::pybind11 INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::pybind11 PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_PYBIND11} ${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PYBIND11_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_PYBIND11_RELEASE} ${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PYBIND11_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_PYBIND11_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PYBIND11_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_PYBIND11_MINSIZEREL} ${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PYBIND11_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_PYBIND11_DEBUG} ${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PYBIND11_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::pybind11 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_PYBIND11}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_PYBIND11_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_PYBIND11_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_PYBIND11_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_PYBIND11_DEBUG}>)
    set_property(TARGET CONAN_PKG::pybind11 PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_PYBIND11}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_PYBIND11_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_PYBIND11_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_PYBIND11_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_PYBIND11_DEBUG}>)
    set_property(TARGET CONAN_PKG::pybind11 PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_PYBIND11_LIST} ${CONAN_CXX_FLAGS_PYBIND11_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_PYBIND11_RELEASE_LIST} ${CONAN_CXX_FLAGS_PYBIND11_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_PYBIND11_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_PYBIND11_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_PYBIND11_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_PYBIND11_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_PYBIND11_DEBUG_LIST}  ${CONAN_CXX_FLAGS_PYBIND11_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES "${CONAN_SYSTEM_LIBS_ABSEIL} ${CONAN_FRAMEWORKS_FOUND_ABSEIL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES "${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ABSEIL}" "${CONAN_LIB_DIRS_ABSEIL}"
                                  CONAN_PACKAGE_TARGETS_ABSEIL "${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES}"
                                  "" abseil)
    set(_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_ABSEIL_DEBUG} ${CONAN_FRAMEWORKS_FOUND_ABSEIL_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ABSEIL_DEBUG}" "${CONAN_LIB_DIRS_ABSEIL_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_ABSEIL_DEBUG "${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_DEBUG}"
                                  "debug" abseil)
    set(_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_ABSEIL_RELEASE} ${CONAN_FRAMEWORKS_FOUND_ABSEIL_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ABSEIL_RELEASE}" "${CONAN_LIB_DIRS_ABSEIL_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_ABSEIL_RELEASE "${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_RELEASE}"
                                  "release" abseil)
    set(_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_ABSEIL_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_ABSEIL_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ABSEIL_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_ABSEIL_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_ABSEIL_RELWITHDEBINFO "${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" abseil)
    set(_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_ABSEIL_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_ABSEIL_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ABSEIL_MINSIZEREL}" "${CONAN_LIB_DIRS_ABSEIL_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_ABSEIL_MINSIZEREL "${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" abseil)

    add_library(CONAN_PKG::abseil INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::abseil PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_ABSEIL} ${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ABSEIL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ABSEIL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ABSEIL_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_ABSEIL_RELEASE} ${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ABSEIL_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ABSEIL_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ABSEIL_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_ABSEIL_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ABSEIL_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ABSEIL_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ABSEIL_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_ABSEIL_MINSIZEREL} ${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ABSEIL_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ABSEIL_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ABSEIL_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_ABSEIL_DEBUG} ${_CONAN_PKG_LIBS_ABSEIL_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ABSEIL_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ABSEIL_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ABSEIL_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::abseil PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_ABSEIL}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_ABSEIL_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_ABSEIL_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_ABSEIL_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_ABSEIL_DEBUG}>)
    set_property(TARGET CONAN_PKG::abseil PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_ABSEIL}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_ABSEIL_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_ABSEIL_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_ABSEIL_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_ABSEIL_DEBUG}>)
    set_property(TARGET CONAN_PKG::abseil PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_ABSEIL_LIST} ${CONAN_CXX_FLAGS_ABSEIL_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_ABSEIL_RELEASE_LIST} ${CONAN_CXX_FLAGS_ABSEIL_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_ABSEIL_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_ABSEIL_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_ABSEIL_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_ABSEIL_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_ABSEIL_DEBUG_LIST}  ${CONAN_CXX_FLAGS_ABSEIL_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES "${CONAN_SYSTEM_LIBS_NETHOST} ${CONAN_FRAMEWORKS_FOUND_NETHOST} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_NETHOST_DEPENDENCIES "${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_NETHOST}" "${CONAN_LIB_DIRS_NETHOST}"
                                  CONAN_PACKAGE_TARGETS_NETHOST "${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES}"
                                  "" nethost)
    set(_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_NETHOST_DEBUG} ${CONAN_FRAMEWORKS_FOUND_NETHOST_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_NETHOST_DEBUG}" "${CONAN_LIB_DIRS_NETHOST_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_NETHOST_DEBUG "${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_DEBUG}"
                                  "debug" nethost)
    set(_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_NETHOST_RELEASE} ${CONAN_FRAMEWORKS_FOUND_NETHOST_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_NETHOST_RELEASE}" "${CONAN_LIB_DIRS_NETHOST_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_NETHOST_RELEASE "${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_RELEASE}"
                                  "release" nethost)
    set(_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_NETHOST_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_NETHOST_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_NETHOST_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_NETHOST_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_NETHOST_RELWITHDEBINFO "${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" nethost)
    set(_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_NETHOST_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_NETHOST_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_NETHOST_MINSIZEREL}" "${CONAN_LIB_DIRS_NETHOST_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_NETHOST_MINSIZEREL "${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" nethost)

    add_library(CONAN_PKG::nethost INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::nethost PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_NETHOST} ${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NETHOST_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NETHOST_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_NETHOST_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_NETHOST_RELEASE} ${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NETHOST_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NETHOST_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_NETHOST_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_NETHOST_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NETHOST_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NETHOST_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_NETHOST_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_NETHOST_MINSIZEREL} ${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NETHOST_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NETHOST_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_NETHOST_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_NETHOST_DEBUG} ${_CONAN_PKG_LIBS_NETHOST_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NETHOST_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NETHOST_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_NETHOST_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::nethost PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_NETHOST}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_NETHOST_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_NETHOST_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_NETHOST_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_NETHOST_DEBUG}>)
    set_property(TARGET CONAN_PKG::nethost PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_NETHOST}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_NETHOST_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_NETHOST_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_NETHOST_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_NETHOST_DEBUG}>)
    set_property(TARGET CONAN_PKG::nethost PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_NETHOST_LIST} ${CONAN_CXX_FLAGS_NETHOST_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_NETHOST_RELEASE_LIST} ${CONAN_CXX_FLAGS_NETHOST_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_NETHOST_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_NETHOST_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_NETHOST_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_NETHOST_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_NETHOST_DEBUG_LIST}  ${CONAN_CXX_FLAGS_NETHOST_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES "${CONAN_SYSTEM_LIBS_MAGIC_ENUM} ${CONAN_FRAMEWORKS_FOUND_MAGIC_ENUM} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES "${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_MAGIC_ENUM}" "${CONAN_LIB_DIRS_MAGIC_ENUM}"
                                  CONAN_PACKAGE_TARGETS_MAGIC_ENUM "${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES}"
                                  "" magic_enum)
    set(_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_MAGIC_ENUM_DEBUG} ${CONAN_FRAMEWORKS_FOUND_MAGIC_ENUM_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_MAGIC_ENUM_DEBUG}" "${CONAN_LIB_DIRS_MAGIC_ENUM_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_MAGIC_ENUM_DEBUG "${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_DEBUG}"
                                  "debug" magic_enum)
    set(_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_MAGIC_ENUM_RELEASE} ${CONAN_FRAMEWORKS_FOUND_MAGIC_ENUM_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_MAGIC_ENUM_RELEASE}" "${CONAN_LIB_DIRS_MAGIC_ENUM_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_MAGIC_ENUM_RELEASE "${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_RELEASE}"
                                  "release" magic_enum)
    set(_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_MAGIC_ENUM_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_MAGIC_ENUM_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_MAGIC_ENUM_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_MAGIC_ENUM_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_MAGIC_ENUM_RELWITHDEBINFO "${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" magic_enum)
    set(_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_MAGIC_ENUM_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_MAGIC_ENUM_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_MAGIC_ENUM_MINSIZEREL}" "${CONAN_LIB_DIRS_MAGIC_ENUM_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_MAGIC_ENUM_MINSIZEREL "${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" magic_enum)

    add_library(CONAN_PKG::magic_enum INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::magic_enum PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_MAGIC_ENUM} ${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_MAGIC_ENUM_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_MAGIC_ENUM_RELEASE} ${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_MAGIC_ENUM_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_MAGIC_ENUM_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_MAGIC_ENUM_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_MAGIC_ENUM_MINSIZEREL} ${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_MAGIC_ENUM_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_MAGIC_ENUM_DEBUG} ${_CONAN_PKG_LIBS_MAGIC_ENUM_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_MAGIC_ENUM_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_MAGIC_ENUM_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::magic_enum PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_MAGIC_ENUM}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_MAGIC_ENUM_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_MAGIC_ENUM_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_MAGIC_ENUM_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_MAGIC_ENUM_DEBUG}>)
    set_property(TARGET CONAN_PKG::magic_enum PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_MAGIC_ENUM}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_MAGIC_ENUM_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_MAGIC_ENUM_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_MAGIC_ENUM_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_MAGIC_ENUM_DEBUG}>)
    set_property(TARGET CONAN_PKG::magic_enum PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_MAGIC_ENUM_LIST} ${CONAN_CXX_FLAGS_MAGIC_ENUM_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_MAGIC_ENUM_RELEASE_LIST} ${CONAN_CXX_FLAGS_MAGIC_ENUM_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_MAGIC_ENUM_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_MAGIC_ENUM_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_MAGIC_ENUM_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_MAGIC_ENUM_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_MAGIC_ENUM_DEBUG_LIST}  ${CONAN_CXX_FLAGS_MAGIC_ENUM_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES "${CONAN_SYSTEM_LIBS_SPDLOG} ${CONAN_FRAMEWORKS_FOUND_SPDLOG} CONAN_PKG::fmt")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_SPDLOG}" "${CONAN_LIB_DIRS_SPDLOG}"
                                  CONAN_PACKAGE_TARGETS_SPDLOG "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES}"
                                  "" spdlog)
    set(_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_SPDLOG_DEBUG} ${CONAN_FRAMEWORKS_FOUND_SPDLOG_DEBUG} CONAN_PKG::fmt")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_SPDLOG_DEBUG}" "${CONAN_LIB_DIRS_SPDLOG_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_SPDLOG_DEBUG "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_DEBUG}"
                                  "debug" spdlog)
    set(_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_SPDLOG_RELEASE} ${CONAN_FRAMEWORKS_FOUND_SPDLOG_RELEASE} CONAN_PKG::fmt")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_SPDLOG_RELEASE}" "${CONAN_LIB_DIRS_SPDLOG_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_SPDLOG_RELEASE "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELEASE}"
                                  "release" spdlog)
    set(_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_SPDLOG_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_SPDLOG_RELWITHDEBINFO} CONAN_PKG::fmt")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_SPDLOG_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_SPDLOG_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_SPDLOG_RELWITHDEBINFO "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" spdlog)
    set(_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_SPDLOG_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_SPDLOG_MINSIZEREL} CONAN_PKG::fmt")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_SPDLOG_MINSIZEREL}" "${CONAN_LIB_DIRS_SPDLOG_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_SPDLOG_MINSIZEREL "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" spdlog)

    add_library(CONAN_PKG::spdlog INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::spdlog PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_SPDLOG} ${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_SPDLOG_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_SPDLOG_RELEASE} ${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_SPDLOG_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_SPDLOG_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_SPDLOG_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_SPDLOG_MINSIZEREL} ${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_SPDLOG_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_SPDLOG_DEBUG} ${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_SPDLOG_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::spdlog PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_SPDLOG}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_SPDLOG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_SPDLOG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_SPDLOG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_SPDLOG_DEBUG}>)
    set_property(TARGET CONAN_PKG::spdlog PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_SPDLOG}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_SPDLOG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_SPDLOG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_SPDLOG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_SPDLOG_DEBUG}>)
    set_property(TARGET CONAN_PKG::spdlog PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_SPDLOG_LIST} ${CONAN_CXX_FLAGS_SPDLOG_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_SPDLOG_RELEASE_LIST} ${CONAN_CXX_FLAGS_SPDLOG_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_SPDLOG_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_SPDLOG_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_SPDLOG_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_SPDLOG_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_SPDLOG_DEBUG_LIST}  ${CONAN_CXX_FLAGS_SPDLOG_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_INJA_DEPENDENCIES "${CONAN_SYSTEM_LIBS_INJA} ${CONAN_FRAMEWORKS_FOUND_INJA} CONAN_PKG::nlohmann_json")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_INJA_DEPENDENCIES "${_CONAN_PKG_LIBS_INJA_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_INJA}" "${CONAN_LIB_DIRS_INJA}"
                                  CONAN_PACKAGE_TARGETS_INJA "${_CONAN_PKG_LIBS_INJA_DEPENDENCIES}"
                                  "" inja)
    set(_CONAN_PKG_LIBS_INJA_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_INJA_DEBUG} ${CONAN_FRAMEWORKS_FOUND_INJA_DEBUG} CONAN_PKG::nlohmann_json")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_INJA_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_INJA_DEBUG}" "${CONAN_LIB_DIRS_INJA_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_INJA_DEBUG "${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_DEBUG}"
                                  "debug" inja)
    set(_CONAN_PKG_LIBS_INJA_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_INJA_RELEASE} ${CONAN_FRAMEWORKS_FOUND_INJA_RELEASE} CONAN_PKG::nlohmann_json")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_INJA_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_INJA_RELEASE}" "${CONAN_LIB_DIRS_INJA_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_INJA_RELEASE "${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_RELEASE}"
                                  "release" inja)
    set(_CONAN_PKG_LIBS_INJA_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_INJA_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_INJA_RELWITHDEBINFO} CONAN_PKG::nlohmann_json")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_INJA_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_INJA_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_INJA_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_INJA_RELWITHDEBINFO "${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" inja)
    set(_CONAN_PKG_LIBS_INJA_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_INJA_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_INJA_MINSIZEREL} CONAN_PKG::nlohmann_json")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_INJA_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_INJA_MINSIZEREL}" "${CONAN_LIB_DIRS_INJA_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_INJA_MINSIZEREL "${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" inja)

    add_library(CONAN_PKG::inja INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::inja PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_INJA} ${_CONAN_PKG_LIBS_INJA_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_INJA_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_INJA_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_INJA_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_INJA_RELEASE} ${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_INJA_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_INJA_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_INJA_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_INJA_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_INJA_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_INJA_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_INJA_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_INJA_MINSIZEREL} ${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_INJA_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_INJA_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_INJA_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_INJA_DEBUG} ${_CONAN_PKG_LIBS_INJA_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_INJA_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_INJA_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_INJA_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::inja PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_INJA}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_INJA_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_INJA_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_INJA_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_INJA_DEBUG}>)
    set_property(TARGET CONAN_PKG::inja PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_INJA}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_INJA_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_INJA_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_INJA_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_INJA_DEBUG}>)
    set_property(TARGET CONAN_PKG::inja PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_INJA_LIST} ${CONAN_CXX_FLAGS_INJA_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_INJA_RELEASE_LIST} ${CONAN_CXX_FLAGS_INJA_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_INJA_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_INJA_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_INJA_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_INJA_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_INJA_DEBUG_LIST}  ${CONAN_CXX_FLAGS_INJA_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES "${CONAN_SYSTEM_LIBS_VULKAN-LOADER} ${CONAN_FRAMEWORKS_FOUND_VULKAN-LOADER} CONAN_PKG::vulkan-headers")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES "${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VULKAN-LOADER}" "${CONAN_LIB_DIRS_VULKAN-LOADER}"
                                  CONAN_PACKAGE_TARGETS_VULKAN-LOADER "${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES}"
                                  "" vulkan-loader)
    set(_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_VULKAN-LOADER_DEBUG} ${CONAN_FRAMEWORKS_FOUND_VULKAN-LOADER_DEBUG} CONAN_PKG::vulkan-headers")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VULKAN-LOADER_DEBUG}" "${CONAN_LIB_DIRS_VULKAN-LOADER_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_VULKAN-LOADER_DEBUG "${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_DEBUG}"
                                  "debug" vulkan-loader)
    set(_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_VULKAN-LOADER_RELEASE} ${CONAN_FRAMEWORKS_FOUND_VULKAN-LOADER_RELEASE} CONAN_PKG::vulkan-headers")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VULKAN-LOADER_RELEASE}" "${CONAN_LIB_DIRS_VULKAN-LOADER_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_VULKAN-LOADER_RELEASE "${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_RELEASE}"
                                  "release" vulkan-loader)
    set(_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_VULKAN-LOADER_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_VULKAN-LOADER_RELWITHDEBINFO} CONAN_PKG::vulkan-headers")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VULKAN-LOADER_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_VULKAN-LOADER_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_VULKAN-LOADER_RELWITHDEBINFO "${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" vulkan-loader)
    set(_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_VULKAN-LOADER_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_VULKAN-LOADER_MINSIZEREL} CONAN_PKG::vulkan-headers")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VULKAN-LOADER_MINSIZEREL}" "${CONAN_LIB_DIRS_VULKAN-LOADER_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_VULKAN-LOADER_MINSIZEREL "${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" vulkan-loader)

    add_library(CONAN_PKG::vulkan-loader INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::vulkan-loader PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_VULKAN-LOADER} ${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VULKAN-LOADER_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_VULKAN-LOADER_RELEASE} ${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VULKAN-LOADER_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_VULKAN-LOADER_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VULKAN-LOADER_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_VULKAN-LOADER_MINSIZEREL} ${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VULKAN-LOADER_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_VULKAN-LOADER_DEBUG} ${_CONAN_PKG_LIBS_VULKAN-LOADER_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-LOADER_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VULKAN-LOADER_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::vulkan-loader PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_VULKAN-LOADER}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_VULKAN-LOADER_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_VULKAN-LOADER_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_VULKAN-LOADER_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_VULKAN-LOADER_DEBUG}>)
    set_property(TARGET CONAN_PKG::vulkan-loader PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_VULKAN-LOADER}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_VULKAN-LOADER_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_VULKAN-LOADER_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_VULKAN-LOADER_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_VULKAN-LOADER_DEBUG}>)
    set_property(TARGET CONAN_PKG::vulkan-loader PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_VULKAN-LOADER_LIST} ${CONAN_CXX_FLAGS_VULKAN-LOADER_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_VULKAN-LOADER_RELEASE_LIST} ${CONAN_CXX_FLAGS_VULKAN-LOADER_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_VULKAN-LOADER_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_VULKAN-LOADER_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_VULKAN-LOADER_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_VULKAN-LOADER_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_VULKAN-LOADER_DEBUG_LIST}  ${CONAN_CXX_FLAGS_VULKAN-LOADER_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_FMT_DEPENDENCIES "${CONAN_SYSTEM_LIBS_FMT} ${CONAN_FRAMEWORKS_FOUND_FMT} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FMT_DEPENDENCIES "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FMT}" "${CONAN_LIB_DIRS_FMT}"
                                  CONAN_PACKAGE_TARGETS_FMT "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES}"
                                  "" fmt)
    set(_CONAN_PKG_LIBS_FMT_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_FMT_DEBUG} ${CONAN_FRAMEWORKS_FOUND_FMT_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FMT_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FMT_DEBUG}" "${CONAN_LIB_DIRS_FMT_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_FMT_DEBUG "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_DEBUG}"
                                  "debug" fmt)
    set(_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_FMT_RELEASE} ${CONAN_FRAMEWORKS_FOUND_FMT_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FMT_RELEASE}" "${CONAN_LIB_DIRS_FMT_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_FMT_RELEASE "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELEASE}"
                                  "release" fmt)
    set(_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_FMT_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_FMT_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FMT_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_FMT_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_FMT_RELWITHDEBINFO "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" fmt)
    set(_CONAN_PKG_LIBS_FMT_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_FMT_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_FMT_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FMT_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FMT_MINSIZEREL}" "${CONAN_LIB_DIRS_FMT_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_FMT_MINSIZEREL "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" fmt)

    add_library(CONAN_PKG::fmt INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::fmt PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_FMT} ${_CONAN_PKG_LIBS_FMT_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FMT_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_FMT_RELEASE} ${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FMT_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_FMT_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FMT_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_FMT_MINSIZEREL} ${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FMT_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_FMT_DEBUG} ${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FMT_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::fmt PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_FMT}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_FMT_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_FMT_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_FMT_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_FMT_DEBUG}>)
    set_property(TARGET CONAN_PKG::fmt PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_FMT}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_FMT_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_FMT_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_FMT_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_FMT_DEBUG}>)
    set_property(TARGET CONAN_PKG::fmt PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_FMT_LIST} ${CONAN_CXX_FLAGS_FMT_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_FMT_RELEASE_LIST} ${CONAN_CXX_FLAGS_FMT_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_FMT_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_FMT_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_FMT_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_FMT_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_FMT_DEBUG_LIST}  ${CONAN_CXX_FLAGS_FMT_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES "${CONAN_SYSTEM_LIBS_NLOHMANN_JSON} ${CONAN_FRAMEWORKS_FOUND_NLOHMANN_JSON} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES "${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_NLOHMANN_JSON}" "${CONAN_LIB_DIRS_NLOHMANN_JSON}"
                                  CONAN_PACKAGE_TARGETS_NLOHMANN_JSON "${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES}"
                                  "" nlohmann_json)
    set(_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_NLOHMANN_JSON_DEBUG} ${CONAN_FRAMEWORKS_FOUND_NLOHMANN_JSON_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_NLOHMANN_JSON_DEBUG}" "${CONAN_LIB_DIRS_NLOHMANN_JSON_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_NLOHMANN_JSON_DEBUG "${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_DEBUG}"
                                  "debug" nlohmann_json)
    set(_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_NLOHMANN_JSON_RELEASE} ${CONAN_FRAMEWORKS_FOUND_NLOHMANN_JSON_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_NLOHMANN_JSON_RELEASE}" "${CONAN_LIB_DIRS_NLOHMANN_JSON_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_NLOHMANN_JSON_RELEASE "${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_RELEASE}"
                                  "release" nlohmann_json)
    set(_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_NLOHMANN_JSON_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_NLOHMANN_JSON_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_NLOHMANN_JSON_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_NLOHMANN_JSON_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_NLOHMANN_JSON_RELWITHDEBINFO "${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" nlohmann_json)
    set(_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_NLOHMANN_JSON_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_NLOHMANN_JSON_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_NLOHMANN_JSON_MINSIZEREL}" "${CONAN_LIB_DIRS_NLOHMANN_JSON_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_NLOHMANN_JSON_MINSIZEREL "${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" nlohmann_json)

    add_library(CONAN_PKG::nlohmann_json INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::nlohmann_json PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_NLOHMANN_JSON} ${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_NLOHMANN_JSON_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_NLOHMANN_JSON_RELEASE} ${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_NLOHMANN_JSON_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_NLOHMANN_JSON_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_NLOHMANN_JSON_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_NLOHMANN_JSON_MINSIZEREL} ${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_NLOHMANN_JSON_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_NLOHMANN_JSON_DEBUG} ${_CONAN_PKG_LIBS_NLOHMANN_JSON_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_NLOHMANN_JSON_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_NLOHMANN_JSON_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::nlohmann_json PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_NLOHMANN_JSON}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_NLOHMANN_JSON_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_NLOHMANN_JSON_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_NLOHMANN_JSON_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_NLOHMANN_JSON_DEBUG}>)
    set_property(TARGET CONAN_PKG::nlohmann_json PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_NLOHMANN_JSON}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_NLOHMANN_JSON_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_NLOHMANN_JSON_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_NLOHMANN_JSON_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_NLOHMANN_JSON_DEBUG}>)
    set_property(TARGET CONAN_PKG::nlohmann_json PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_NLOHMANN_JSON_LIST} ${CONAN_CXX_FLAGS_NLOHMANN_JSON_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_NLOHMANN_JSON_RELEASE_LIST} ${CONAN_CXX_FLAGS_NLOHMANN_JSON_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_NLOHMANN_JSON_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_NLOHMANN_JSON_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_NLOHMANN_JSON_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_NLOHMANN_JSON_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_NLOHMANN_JSON_DEBUG_LIST}  ${CONAN_CXX_FLAGS_NLOHMANN_JSON_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES "${CONAN_SYSTEM_LIBS_VULKAN-HEADERS} ${CONAN_FRAMEWORKS_FOUND_VULKAN-HEADERS} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES "${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VULKAN-HEADERS}" "${CONAN_LIB_DIRS_VULKAN-HEADERS}"
                                  CONAN_PACKAGE_TARGETS_VULKAN-HEADERS "${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES}"
                                  "" vulkan-headers)
    set(_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_VULKAN-HEADERS_DEBUG} ${CONAN_FRAMEWORKS_FOUND_VULKAN-HEADERS_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VULKAN-HEADERS_DEBUG}" "${CONAN_LIB_DIRS_VULKAN-HEADERS_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_VULKAN-HEADERS_DEBUG "${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_DEBUG}"
                                  "debug" vulkan-headers)
    set(_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_VULKAN-HEADERS_RELEASE} ${CONAN_FRAMEWORKS_FOUND_VULKAN-HEADERS_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VULKAN-HEADERS_RELEASE}" "${CONAN_LIB_DIRS_VULKAN-HEADERS_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_VULKAN-HEADERS_RELEASE "${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_RELEASE}"
                                  "release" vulkan-headers)
    set(_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_VULKAN-HEADERS_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_VULKAN-HEADERS_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VULKAN-HEADERS_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_VULKAN-HEADERS_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_VULKAN-HEADERS_RELWITHDEBINFO "${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" vulkan-headers)
    set(_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_VULKAN-HEADERS_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_VULKAN-HEADERS_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VULKAN-HEADERS_MINSIZEREL}" "${CONAN_LIB_DIRS_VULKAN-HEADERS_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_VULKAN-HEADERS_MINSIZEREL "${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" vulkan-headers)

    add_library(CONAN_PKG::vulkan-headers INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::vulkan-headers PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_VULKAN-HEADERS} ${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VULKAN-HEADERS_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_VULKAN-HEADERS_RELEASE} ${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VULKAN-HEADERS_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_VULKAN-HEADERS_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VULKAN-HEADERS_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_VULKAN-HEADERS_MINSIZEREL} ${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VULKAN-HEADERS_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_VULKAN-HEADERS_DEBUG} ${_CONAN_PKG_LIBS_VULKAN-HEADERS_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VULKAN-HEADERS_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VULKAN-HEADERS_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::vulkan-headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_VULKAN-HEADERS}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_VULKAN-HEADERS_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_VULKAN-HEADERS_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_VULKAN-HEADERS_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_VULKAN-HEADERS_DEBUG}>)
    set_property(TARGET CONAN_PKG::vulkan-headers PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_VULKAN-HEADERS}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_VULKAN-HEADERS_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_VULKAN-HEADERS_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_VULKAN-HEADERS_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_VULKAN-HEADERS_DEBUG}>)
    set_property(TARGET CONAN_PKG::vulkan-headers PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_VULKAN-HEADERS_LIST} ${CONAN_CXX_FLAGS_VULKAN-HEADERS_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_VULKAN-HEADERS_RELEASE_LIST} ${CONAN_CXX_FLAGS_VULKAN-HEADERS_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_VULKAN-HEADERS_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_VULKAN-HEADERS_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_VULKAN-HEADERS_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_VULKAN-HEADERS_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_VULKAN-HEADERS_DEBUG_LIST}  ${CONAN_CXX_FLAGS_VULKAN-HEADERS_DEBUG_LIST}>)

    set(CONAN_TARGETS CONAN_PKG::gsl-lite CONAN_PKG::mpark-variant CONAN_PKG::hkg CONAN_PKG::pybind11 CONAN_PKG::abseil CONAN_PKG::nethost CONAN_PKG::magic_enum CONAN_PKG::spdlog CONAN_PKG::inja CONAN_PKG::vulkan-loader CONAN_PKG::fmt CONAN_PKG::nlohmann_json CONAN_PKG::vulkan-headers)

endmacro()


macro(conan_basic_setup)
    set(options TARGETS NO_OUTPUT_DIRS SKIP_RPATH KEEP_RPATHS SKIP_STD SKIP_FPIC)
    cmake_parse_arguments(ARGUMENTS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if(CONAN_EXPORTED)
        conan_message(STATUS "Conan: called by CMake conan helper")
    endif()

    if(CONAN_IN_LOCAL_CACHE)
        conan_message(STATUS "Conan: called inside local cache")
    endif()

    if(NOT ARGUMENTS_NO_OUTPUT_DIRS)
        conan_message(STATUS "Conan: Adjusting output directories")
        conan_output_dirs_setup()
    endif()

    if(NOT ARGUMENTS_TARGETS)
        conan_message(STATUS "Conan: Using cmake global configuration")
        conan_global_flags()
    else()
        conan_message(STATUS "Conan: Using cmake targets configuration")
        conan_define_targets()
    endif()

    if(ARGUMENTS_SKIP_RPATH)
        # Change by "DEPRECATION" or "SEND_ERROR" when we are ready
        conan_message(WARNING "Conan: SKIP_RPATH is deprecated, it has been renamed to KEEP_RPATHS")
    endif()

    if(NOT ARGUMENTS_SKIP_RPATH AND NOT ARGUMENTS_KEEP_RPATHS)
        # Parameter has renamed, but we keep the compatibility with old SKIP_RPATH
        conan_set_rpath()
    endif()

    if(NOT ARGUMENTS_SKIP_STD)
        conan_set_std()
    endif()

    if(NOT ARGUMENTS_SKIP_FPIC)
        conan_set_fpic()
    endif()

    conan_check_compiler()
    conan_set_libcxx()
    conan_set_vs_runtime()
    conan_set_find_paths()
    conan_include_build_modules()
    conan_set_find_library_paths()
endmacro()


macro(conan_set_find_paths)
    # CMAKE_MODULE_PATH does not have Debug/Release config, but there are variables
    # CONAN_CMAKE_MODULE_PATH_DEBUG to be used by the consumer
    # CMake can find findXXX.cmake files in the root of packages
    set(CMAKE_MODULE_PATH ${CONAN_CMAKE_MODULE_PATH} ${CMAKE_MODULE_PATH})

    # Make find_package() to work
    set(CMAKE_PREFIX_PATH ${CONAN_CMAKE_MODULE_PATH} ${CMAKE_PREFIX_PATH})

    # Set the find root path (cross build)
    set(CMAKE_FIND_ROOT_PATH ${CONAN_CMAKE_FIND_ROOT_PATH} ${CMAKE_FIND_ROOT_PATH})
    if(CONAN_CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
        set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ${CONAN_CMAKE_FIND_ROOT_PATH_MODE_PROGRAM})
    endif()
    if(CONAN_CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
        set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ${CONAN_CMAKE_FIND_ROOT_PATH_MODE_LIBRARY})
    endif()
    if(CONAN_CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
        set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ${CONAN_CMAKE_FIND_ROOT_PATH_MODE_INCLUDE})
    endif()
endmacro()


macro(conan_set_find_library_paths)
    # CMAKE_INCLUDE_PATH, CMAKE_LIBRARY_PATH does not have Debug/Release config, but there are variables
    # CONAN_INCLUDE_DIRS_DEBUG/RELEASE CONAN_LIB_DIRS_DEBUG/RELEASE to be used by the consumer
    # For find_library
    set(CMAKE_INCLUDE_PATH ${CONAN_INCLUDE_DIRS} ${CMAKE_INCLUDE_PATH})
    set(CMAKE_LIBRARY_PATH ${CONAN_LIB_DIRS} ${CMAKE_LIBRARY_PATH})
endmacro()


macro(conan_set_vs_runtime)
    if(CONAN_LINK_RUNTIME)
        conan_get_policy(CMP0091 policy_0091)
        if(policy_0091 STREQUAL "NEW")
            if(CONAN_LINK_RUNTIME MATCHES "MTd")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebug")
            elseif(CONAN_LINK_RUNTIME MATCHES "MDd")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
            elseif(CONAN_LINK_RUNTIME MATCHES "MT")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
            elseif(CONAN_LINK_RUNTIME MATCHES "MD")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
            endif()
        else()
            foreach(flag CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE
                         CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_RELWITHDEBINFO
                         CMAKE_C_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_MINSIZEREL)
                if(DEFINED ${flag})
                    string(REPLACE "/MD" ${CONAN_LINK_RUNTIME} ${flag} "${${flag}}")
                endif()
            endforeach()
            foreach(flag CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS_DEBUG)
                if(DEFINED ${flag})
                    string(REPLACE "/MDd" ${CONAN_LINK_RUNTIME} ${flag} "${${flag}}")
                endif()
            endforeach()
        endif()
    endif()
endmacro()


macro(conan_flags_setup)
    # Macro maintained for backwards compatibility
    conan_set_find_library_paths()
    conan_global_flags()
    conan_set_rpath()
    conan_set_vs_runtime()
    conan_set_libcxx()
endmacro()


function(conan_message MESSAGE_OUTPUT)
    if(NOT CONAN_CMAKE_SILENT_OUTPUT)
        message(${ARGV${0}})
    endif()
endfunction()


function(conan_get_policy policy_id policy)
    if(POLICY "${policy_id}")
        cmake_policy(GET "${policy_id}" _policy)
        set(${policy} "${_policy}" PARENT_SCOPE)
    else()
        set(${policy} "" PARENT_SCOPE)
    endif()
endfunction()


function(conan_find_libraries_abs_path libraries package_libdir libraries_abs_path)
    foreach(_LIBRARY_NAME ${libraries})
        find_library(CONAN_FOUND_LIBRARY NAMES ${_LIBRARY_NAME} PATHS ${package_libdir}
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        if(CONAN_FOUND_LIBRARY)
            conan_message(STATUS "Library ${_LIBRARY_NAME} found ${CONAN_FOUND_LIBRARY}")
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${CONAN_FOUND_LIBRARY})
        else()
            conan_message(STATUS "Library ${_LIBRARY_NAME} not found in package, might be system one")
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${_LIBRARY_NAME})
        endif()
        unset(CONAN_FOUND_LIBRARY CACHE)
    endforeach()
    set(${libraries_abs_path} ${CONAN_FULLPATH_LIBS} PARENT_SCOPE)
endfunction()


function(conan_package_library_targets libraries package_libdir libraries_abs_path deps build_type package_name)
    unset(_CONAN_ACTUAL_TARGETS CACHE)
    unset(_CONAN_FOUND_SYSTEM_LIBS CACHE)
    foreach(_LIBRARY_NAME ${libraries})
        find_library(CONAN_FOUND_LIBRARY NAMES ${_LIBRARY_NAME} PATHS ${package_libdir}
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        if(CONAN_FOUND_LIBRARY)
            conan_message(STATUS "Library ${_LIBRARY_NAME} found ${CONAN_FOUND_LIBRARY}")
            set(_LIB_NAME CONAN_LIB::${package_name}_${_LIBRARY_NAME}${build_type})
            add_library(${_LIB_NAME} UNKNOWN IMPORTED)
            set_target_properties(${_LIB_NAME} PROPERTIES IMPORTED_LOCATION ${CONAN_FOUND_LIBRARY})
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${_LIB_NAME})
            set(_CONAN_ACTUAL_TARGETS ${_CONAN_ACTUAL_TARGETS} ${_LIB_NAME})
        else()
            conan_message(STATUS "Library ${_LIBRARY_NAME} not found in package, might be system one")
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${_LIBRARY_NAME})
            set(_CONAN_FOUND_SYSTEM_LIBS "${_CONAN_FOUND_SYSTEM_LIBS};${_LIBRARY_NAME}")
        endif()
        unset(CONAN_FOUND_LIBRARY CACHE)
    endforeach()

    # Add all dependencies to all targets
    string(REPLACE " " ";" deps_list "${deps}")
    foreach(_CONAN_ACTUAL_TARGET ${_CONAN_ACTUAL_TARGETS})
        set_property(TARGET ${_CONAN_ACTUAL_TARGET} PROPERTY INTERFACE_LINK_LIBRARIES "${_CONAN_FOUND_SYSTEM_LIBS};${deps_list}")
    endforeach()

    set(${libraries_abs_path} ${CONAN_FULLPATH_LIBS} PARENT_SCOPE)
endfunction()


macro(conan_set_libcxx)
    if(DEFINED CONAN_LIBCXX)
        conan_message(STATUS "Conan: C++ stdlib: ${CONAN_LIBCXX}")
        if(CONAN_COMPILER STREQUAL "clang" OR CONAN_COMPILER STREQUAL "apple-clang")
            if(CONAN_LIBCXX STREQUAL "libstdc++" OR CONAN_LIBCXX STREQUAL "libstdc++11" )
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
            elseif(CONAN_LIBCXX STREQUAL "libc++")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
            endif()
        endif()
        if(CONAN_COMPILER STREQUAL "sun-cc")
            if(CONAN_LIBCXX STREQUAL "libCstd")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=Cstd")
            elseif(CONAN_LIBCXX STREQUAL "libstdcxx")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=stdcxx4")
            elseif(CONAN_LIBCXX STREQUAL "libstlport")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=stlport4")
            elseif(CONAN_LIBCXX STREQUAL "libstdc++")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=stdcpp")
            endif()
        endif()
        if(CONAN_LIBCXX STREQUAL "libstdc++11")
            add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
        elseif(CONAN_LIBCXX STREQUAL "libstdc++")
            add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        endif()
    endif()
endmacro()


macro(conan_set_std)
    conan_message(STATUS "Conan: Adjusting language standard")
    # Do not warn "Manually-specified variables were not used by the project"
    set(ignorevar "${CONAN_STD_CXX_FLAG}${CONAN_CMAKE_CXX_STANDARD}${CONAN_CMAKE_CXX_EXTENSIONS}")
    if (CMAKE_VERSION VERSION_LESS "3.1" OR
        (CMAKE_VERSION VERSION_LESS "3.12" AND ("${CONAN_CMAKE_CXX_STANDARD}" STREQUAL "20" OR "${CONAN_CMAKE_CXX_STANDARD}" STREQUAL "gnu20")))
        if(CONAN_STD_CXX_FLAG)
            conan_message(STATUS "Conan setting CXX_FLAGS flags: ${CONAN_STD_CXX_FLAG}")
            set(CMAKE_CXX_FLAGS "${CONAN_STD_CXX_FLAG} ${CMAKE_CXX_FLAGS}")
        endif()
    else()
        if(CONAN_CMAKE_CXX_STANDARD)
            conan_message(STATUS "Conan setting CPP STANDARD: ${CONAN_CMAKE_CXX_STANDARD} WITH EXTENSIONS ${CONAN_CMAKE_CXX_EXTENSIONS}")
            set(CMAKE_CXX_STANDARD ${CONAN_CMAKE_CXX_STANDARD})
            set(CMAKE_CXX_EXTENSIONS ${CONAN_CMAKE_CXX_EXTENSIONS})
        endif()
    endif()
endmacro()


macro(conan_set_rpath)
    conan_message(STATUS "Conan: Adjusting default RPATHs Conan policies")
    if(APPLE)
        # https://cmake.org/Wiki/CMake_RPATH_handling
        # CONAN GUIDE: All generated libraries should have the id and dependencies to other
        # dylibs without path, just the name, EX:
        # libMyLib1.dylib:
        #     libMyLib1.dylib (compatibility version 0.0.0, current version 0.0.0)
        #     libMyLib0.dylib (compatibility version 0.0.0, current version 0.0.0)
        #     /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        #     /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)
        # AVOID RPATH FOR *.dylib, ALL LIBS BETWEEN THEM AND THE EXE
        # SHOULD BE ON THE LINKER RESOLVER PATH (./ IS ONE OF THEM)
        set(CMAKE_SKIP_RPATH 1 CACHE BOOL "rpaths" FORCE)
        # Policy CMP0068
        # We want the old behavior, in CMake >= 3.9 CMAKE_SKIP_RPATH won't affect the install_name in OSX
        set(CMAKE_INSTALL_NAME_DIR "")
    endif()
endmacro()


macro(conan_set_fpic)
    if(DEFINED CONAN_CMAKE_POSITION_INDEPENDENT_CODE)
        conan_message(STATUS "Conan: Adjusting fPIC flag (${CONAN_CMAKE_POSITION_INDEPENDENT_CODE})")
        set(CMAKE_POSITION_INDEPENDENT_CODE ${CONAN_CMAKE_POSITION_INDEPENDENT_CODE})
    endif()
endmacro()


macro(conan_output_dirs_setup)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endmacro()


macro(conan_split_version VERSION_STRING MAJOR MINOR)
    #make a list from the version string
    string(REPLACE "." ";" VERSION_LIST "${VERSION_STRING}")

    #write output values
    list(LENGTH VERSION_LIST _version_len)
    list(GET VERSION_LIST 0 ${MAJOR})
    if(${_version_len} GREATER 1)
        list(GET VERSION_LIST 1 ${MINOR})
    endif()
endmacro()


macro(conan_error_compiler_version)
    message(FATAL_ERROR "Detected a mismatch for the compiler version between your conan profile settings and CMake: \n"
                        "Compiler version specified in your conan profile: ${CONAN_COMPILER_VERSION}\n"
                        "Compiler version detected in CMake: ${VERSION_MAJOR}.${VERSION_MINOR}\n"
                        "Please check your conan profile settings (conan profile show [default|your_profile_name])\n"
                        "P.S. You may set CONAN_DISABLE_CHECK_COMPILER CMake variable in order to disable this check."
           )
endmacro()

set(_CONAN_CURRENT_DIR ${CMAKE_CURRENT_LIST_DIR})

function(conan_get_compiler CONAN_INFO_COMPILER CONAN_INFO_COMPILER_VERSION)
    conan_message(STATUS "Current conanbuildinfo.cmake directory: " ${_CONAN_CURRENT_DIR})
    if(NOT EXISTS ${_CONAN_CURRENT_DIR}/conaninfo.txt)
        conan_message(STATUS "WARN: conaninfo.txt not found")
        return()
    endif()

    file (READ "${_CONAN_CURRENT_DIR}/conaninfo.txt" CONANINFO)

    # MATCHALL will match all, including the last one, which is the full_settings one
    string(REGEX MATCH "full_settings.*" _FULL_SETTINGS_MATCHED ${CONANINFO})
    string(REGEX MATCH "compiler=([-A-Za-z0-9_ ]+)" _MATCHED ${_FULL_SETTINGS_MATCHED})
    if(DEFINED CMAKE_MATCH_1)
        string(STRIP "${CMAKE_MATCH_1}" _CONAN_INFO_COMPILER)
        set(${CONAN_INFO_COMPILER} ${_CONAN_INFO_COMPILER} PARENT_SCOPE)
    endif()

    string(REGEX MATCH "compiler.version=([-A-Za-z0-9_.]+)" _MATCHED ${_FULL_SETTINGS_MATCHED})
    if(DEFINED CMAKE_MATCH_1)
        string(STRIP "${CMAKE_MATCH_1}" _CONAN_INFO_COMPILER_VERSION)
        set(${CONAN_INFO_COMPILER_VERSION} ${_CONAN_INFO_COMPILER_VERSION} PARENT_SCOPE)
    endif()
endfunction()


function(check_compiler_version)
    conan_split_version(${CMAKE_CXX_COMPILER_VERSION} VERSION_MAJOR VERSION_MINOR)
    if(DEFINED CONAN_SETTINGS_COMPILER_TOOLSET)
       conan_message(STATUS "Conan: Skipping compiler check: Declared 'compiler.toolset'")
       return()
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        # MSVC_VERSION is defined since 2.8.2 at least
        # https://cmake.org/cmake/help/v2.8.2/cmake.html#variable:MSVC_VERSION
        # https://cmake.org/cmake/help/v3.14/variable/MSVC_VERSION.html
        if(
            # 1930 = VS 17.0 (v143 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "17" AND NOT((MSVC_VERSION EQUAL 1930) OR (MSVC_VERSION GREATER 1930))) OR
            # 1920-1929 = VS 16.0 (v142 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "16" AND NOT((MSVC_VERSION GREATER 1919) AND (MSVC_VERSION LESS 1930))) OR
            # 1910-1919 = VS 15.0 (v141 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "15" AND NOT((MSVC_VERSION GREATER 1909) AND (MSVC_VERSION LESS 1920))) OR
            # 1900      = VS 14.0 (v140 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "14" AND NOT(MSVC_VERSION EQUAL 1900)) OR
            # 1800      = VS 12.0 (v120 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "12" AND NOT VERSION_MAJOR STREQUAL "18") OR
            # 1700      = VS 11.0 (v110 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "11" AND NOT VERSION_MAJOR STREQUAL "17") OR
            # 1600      = VS 10.0 (v100 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "10" AND NOT VERSION_MAJOR STREQUAL "16") OR
            # 1500      = VS  9.0 (v90 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "9" AND NOT VERSION_MAJOR STREQUAL "15") OR
            # 1400      = VS  8.0 (v80 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "8" AND NOT VERSION_MAJOR STREQUAL "14") OR
            # 1310      = VS  7.1, 1300      = VS  7.0
            (CONAN_COMPILER_VERSION STREQUAL "7" AND NOT VERSION_MAJOR STREQUAL "13") OR
            # 1200      = VS  6.0
            (CONAN_COMPILER_VERSION STREQUAL "6" AND NOT VERSION_MAJOR STREQUAL "12") )
            conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "gcc")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        set(_CHECK_VERSION ${VERSION_MAJOR}.${VERSION_MINOR})
        set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
        if(NOT ${CONAN_COMPILER_VERSION} VERSION_LESS 5.0)
            conan_message(STATUS "Conan: Compiler GCC>=5, checking major version ${CONAN_COMPILER_VERSION}")
            conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
            if("${CONAN_COMPILER_MINOR}" STREQUAL "")
                set(_CHECK_VERSION ${VERSION_MAJOR})
                set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR})
            endif()
        endif()
        conan_message(STATUS "Conan: Checking correct version: ${_CHECK_VERSION}")
        if(NOT ${_CHECK_VERSION} VERSION_EQUAL ${_CONAN_VERSION})
            conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "clang")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        set(_CHECK_VERSION ${VERSION_MAJOR}.${VERSION_MINOR})
        set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
        if(NOT ${CONAN_COMPILER_VERSION} VERSION_LESS 8.0)
            conan_message(STATUS "Conan: Compiler Clang>=8, checking major version ${CONAN_COMPILER_VERSION}")
            if("${CONAN_COMPILER_MINOR}" STREQUAL "")
                set(_CHECK_VERSION ${VERSION_MAJOR})
                set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR})
            endif()
        endif()
        conan_message(STATUS "Conan: Checking correct version: ${_CHECK_VERSION}")
        if(NOT ${_CHECK_VERSION} VERSION_EQUAL ${_CONAN_VERSION})
            conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "apple-clang" OR CONAN_COMPILER STREQUAL "sun-cc" OR CONAN_COMPILER STREQUAL "mcst-lcc")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        if(${CONAN_COMPILER_MAJOR} VERSION_GREATER_EQUAL "13" AND "${CONAN_COMPILER_MINOR}" STREQUAL "" AND ${CONAN_COMPILER_MAJOR} VERSION_EQUAL ${VERSION_MAJOR})
           # This is correct,  13.X is considered 13
        elseif(NOT ${VERSION_MAJOR}.${VERSION_MINOR} VERSION_EQUAL ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
           conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "intel")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        if(NOT ${CONAN_COMPILER_VERSION} VERSION_LESS 19.1)
            if(NOT ${VERSION_MAJOR}.${VERSION_MINOR} VERSION_EQUAL ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
               conan_error_compiler_version()
            endif()
        else()
            if(NOT ${VERSION_MAJOR} VERSION_EQUAL ${CONAN_COMPILER_MAJOR})
               conan_error_compiler_version()
            endif()
        endif()
    else()
        conan_message(STATUS "WARN: Unknown compiler '${CONAN_COMPILER}', skipping the version check...")
    endif()
endfunction()


function(conan_check_compiler)
    if(CONAN_DISABLE_CHECK_COMPILER)
        conan_message(STATUS "WARN: Disabled conan compiler checks")
        return()
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER_ID)
        if(DEFINED CMAKE_C_COMPILER_ID)
            conan_message(STATUS "This project seems to be plain C, using '${CMAKE_C_COMPILER_ID}' compiler")
            set(CMAKE_CXX_COMPILER_ID ${CMAKE_C_COMPILER_ID})
            set(CMAKE_CXX_COMPILER_VERSION ${CMAKE_C_COMPILER_VERSION})
        else()
            message(FATAL_ERROR "This project seems to be plain C, but no compiler defined")
        endif()
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID AND NOT CMAKE_C_COMPILER_ID)
        # This use case happens when compiler is not identified by CMake, but the compilers are there and work
        conan_message(STATUS "*** WARN: CMake was not able to identify a C or C++ compiler ***")
        conan_message(STATUS "*** WARN: Disabling compiler checks. Please make sure your settings match your environment ***")
        return()
    endif()
    if(NOT DEFINED CONAN_COMPILER)
        conan_get_compiler(CONAN_COMPILER CONAN_COMPILER_VERSION)
        if(NOT DEFINED CONAN_COMPILER)
            conan_message(STATUS "WARN: CONAN_COMPILER variable not set, please make sure yourself that "
                          "your compiler and version matches your declared settings")
            return()
        endif()
    endif()

    if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL ${CMAKE_SYSTEM_NAME})
        set(CROSS_BUILDING 1)
    endif()

    # If using VS, verify toolset
    if (CONAN_COMPILER STREQUAL "Visual Studio")
        if (CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "LLVM" OR
            CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "llvm" OR
            CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "clang" OR
            CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "Clang")
            set(EXPECTED_CMAKE_CXX_COMPILER_ID "Clang")
        elseif (CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "Intel")
            set(EXPECTED_CMAKE_CXX_COMPILER_ID "Intel")
        else()
            set(EXPECTED_CMAKE_CXX_COMPILER_ID "MSVC")
        endif()

        if (NOT CMAKE_CXX_COMPILER_ID MATCHES ${EXPECTED_CMAKE_CXX_COMPILER_ID})
            message(FATAL_ERROR "Incorrect '${CONAN_COMPILER}'. Toolset specifies compiler as '${EXPECTED_CMAKE_CXX_COMPILER_ID}' "
                                "but CMake detected '${CMAKE_CXX_COMPILER_ID}'")
        endif()

    # Avoid checks when cross compiling, apple-clang crashes because its APPLE but not apple-clang
    # Actually CMake is detecting "clang" when you are using apple-clang, only if CMP0025 is set to NEW will detect apple-clang
    elseif((CONAN_COMPILER STREQUAL "gcc" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR
        (CONAN_COMPILER STREQUAL "apple-clang" AND NOT CROSS_BUILDING AND (NOT APPLE OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")) OR
        (CONAN_COMPILER STREQUAL "clang" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR
        (CONAN_COMPILER STREQUAL "sun-cc" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "SunPro") )
        message(FATAL_ERROR "Incorrect '${CONAN_COMPILER}', is not the one detected by CMake: '${CMAKE_CXX_COMPILER_ID}'")
    endif()


    if(NOT DEFINED CONAN_COMPILER_VERSION)
        conan_message(STATUS "WARN: CONAN_COMPILER_VERSION variable not set, please make sure yourself "
                             "that your compiler version matches your declared settings")
        return()
    endif()
    check_compiler_version()
endfunction()


macro(conan_set_flags build_type)
    set(CMAKE_CXX_FLAGS${build_type} "${CMAKE_CXX_FLAGS${build_type}} ${CONAN_CXX_FLAGS${build_type}}")
    set(CMAKE_C_FLAGS${build_type} "${CMAKE_C_FLAGS${build_type}} ${CONAN_C_FLAGS${build_type}}")
    set(CMAKE_SHARED_LINKER_FLAGS${build_type} "${CMAKE_SHARED_LINKER_FLAGS${build_type}} ${CONAN_SHARED_LINKER_FLAGS${build_type}}")
    set(CMAKE_EXE_LINKER_FLAGS${build_type} "${CMAKE_EXE_LINKER_FLAGS${build_type}} ${CONAN_EXE_LINKER_FLAGS${build_type}}")
endmacro()


macro(conan_global_flags)
    if(CONAN_SYSTEM_INCLUDES)
        include_directories(SYSTEM ${CONAN_INCLUDE_DIRS}
                                   "$<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_RELEASE}>"
                                   "$<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_RELWITHDEBINFO}>"
                                   "$<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_MINSIZEREL}>"
                                   "$<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_DEBUG}>")
    else()
        include_directories(${CONAN_INCLUDE_DIRS}
                            "$<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_RELEASE}>"
                            "$<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_RELWITHDEBINFO}>"
                            "$<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_MINSIZEREL}>"
                            "$<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_DEBUG}>")
    endif()

    link_directories(${CONAN_LIB_DIRS})

    conan_find_libraries_abs_path("${CONAN_LIBS_DEBUG}" "${CONAN_LIB_DIRS_DEBUG}"
                                  CONAN_LIBS_DEBUG)
    conan_find_libraries_abs_path("${CONAN_LIBS_RELEASE}" "${CONAN_LIB_DIRS_RELEASE}"
                                  CONAN_LIBS_RELEASE)
    conan_find_libraries_abs_path("${CONAN_LIBS_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_RELWITHDEBINFO}"
                                  CONAN_LIBS_RELWITHDEBINFO)
    conan_find_libraries_abs_path("${CONAN_LIBS_MINSIZEREL}" "${CONAN_LIB_DIRS_MINSIZEREL}"
                                  CONAN_LIBS_MINSIZEREL)

    add_compile_options(${CONAN_DEFINES}
                        "$<$<CONFIG:Debug>:${CONAN_DEFINES_DEBUG}>"
                        "$<$<CONFIG:Release>:${CONAN_DEFINES_RELEASE}>"
                        "$<$<CONFIG:RelWithDebInfo>:${CONAN_DEFINES_RELWITHDEBINFO}>"
                        "$<$<CONFIG:MinSizeRel>:${CONAN_DEFINES_MINSIZEREL}>")

    conan_set_flags("")
    conan_set_flags("_RELEASE")
    conan_set_flags("_DEBUG")

endmacro()


macro(conan_target_link_libraries target)
    if(CONAN_TARGETS)
        target_link_libraries(${target} ${CONAN_TARGETS})
    else()
        target_link_libraries(${target} ${CONAN_LIBS})
        foreach(_LIB ${CONAN_LIBS_RELEASE})
            target_link_libraries(${target} optimized ${_LIB})
        endforeach()
        foreach(_LIB ${CONAN_LIBS_DEBUG})
            target_link_libraries(${target} debug ${_LIB})
        endforeach()
    endif()
endmacro()


macro(conan_include_build_modules)
    if(CMAKE_BUILD_TYPE)
        if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_DEBUG} ${CONAN_BUILD_MODULES_PATHS})
        elseif(${CMAKE_BUILD_TYPE} MATCHES "Release")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_RELEASE} ${CONAN_BUILD_MODULES_PATHS})
        elseif(${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_RELWITHDEBINFO} ${CONAN_BUILD_MODULES_PATHS})
        elseif(${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_MINSIZEREL} ${CONAN_BUILD_MODULES_PATHS})
        endif()
    endif()

    foreach(_BUILD_MODULE_PATH ${CONAN_BUILD_MODULES_PATHS})
        include(${_BUILD_MODULE_PATH})
    endforeach()
endmacro()


### Definition of user declared vars (user_info) ###

