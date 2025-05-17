if (ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX REQUIRED)
endif ()

if (NOT BUILDING_RUNTIME)
    find_package(nethost REQUIRED)
    find_package(fmt REQUIRED)
endif ()

find_package(nlohmann_json REQUIRED)

if (BUILD_TESTING)
    find_package(GTest REQUIRED)
endif ()
