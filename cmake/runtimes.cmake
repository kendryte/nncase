message("Enabled Runtimes:")

foreach(NAME ${NNCASE_ENABLED_RUNTIMES})
    message("- ${NAME}")
    set(NNCASE_BUILTIN_RUNTIMES_DECL "${NNCASE_BUILTIN_RUNTIMES_DECL}result<std::unique_ptr<runtime_base>> create_${NAME}_runtime();\n")
    set(NNCASE_BUILTIN_RUNTIMES_REG "${NNCASE_BUILTIN_RUNTIMES_REG}    { to_array(\"${NAME}\"), create_${NAME}_runtime },\n")
endforeach()

configure_file("${CMAKE_CURRENT_LIST_DIR}/../src/runtime/builtin_runtimes.inc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/src/runtime/builtin_runtimes.inc"
)
