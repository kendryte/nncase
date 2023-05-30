#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "nncaseruntime" for configuration "Release"
set_property(TARGET nncaseruntime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(nncaseruntime PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libNncase.Runtime.Native.so"
  IMPORTED_SONAME_RELEASE "libNncase.Runtime.Native.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS nncaseruntime )
list(APPEND _IMPORT_CHECK_FILES_FOR_nncaseruntime "${_IMPORT_PREFIX}/lib/libNncase.Runtime.Native.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
