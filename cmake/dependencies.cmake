if (ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX REQUIRED)
endif ()

if (NOT BUILDING_RUNTIME)
    find_package(nethost REQUIRED)
    find_package(fmt REQUIRED)
    find_package(nlohmann_json REQUIRED)
endif ()

if (BUILD_TESTING)
    find_package(GTest REQUIRED)
endif ()
