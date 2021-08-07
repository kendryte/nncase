function(target_link_kendryte TARGET)
    target_link_libraries(${TARGET} PRIVATE kendryte)
    get_target_property(TARGET_TYPE ${TARGET} TYPE)
    if (TARGET_TYPE STREQUAL "EXECUTABLE")
        get_target_property(TARGET_OUT_NAME ${TARGET} OUTPUT_NAME)
        set(BIN_NAME $<TARGET_FILE_DIR:${TARGET}>/$<TARGET_FILE_PREFIX:${TARGET}>$<TARGET_FILE_PREFIX:${TARGET}>$<TARGET_FILE_BASE_NAME:${TARGET}>.bin)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_OBJCOPY} --output-format=binary $<TARGET_FILE:${TARGET}> ${BIN_NAME}
            DEPENDS ${TARGET}
            COMMENT "Generating .bin file for ${TARGET}...")
        install(FILES ${BIN_NAME} DESTINATION bin)
    endif()
endfunction()
