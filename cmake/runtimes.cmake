message("Enabled Runtimes:")

foreach(NAME ${NNCASE_ENABLED_RUNTIMES})
    message("- ${NAME}")
    set(NNCASE_BUILTIN_RUNTIMES_DECL "${NNCASE_BUILTIN_RUNTIMES_DECL}result<std::unique_ptr<runtime_base>> create_${NAME}_runtime();result<std::vector<std::pair<std::string, runtime_module::custom_call_type>>>
    create_${NAME}_custom_calls();\n")
    set(NNCASE_BUILTIN_RUNTIMES_REG "${NNCASE_BUILTIN_RUNTIMES_REG}    { to_target_id(\"${NAME}\"), create_${NAME}_runtime , create_${NAME}_custom_calls },\n")
endforeach()

configure_file("${CMAKE_CURRENT_LIST_DIR}/../src/runtime/builtin_runtimes.inl.in"
  "${CMAKE_CURRENT_BINARY_DIR}/src/runtime/builtin_runtimes.inl"
)
