cmake_minimum_required(VERSION 3.10)
project(GRBPairHalo
    VERSION 1.0
    LANGUAGES CXX
)

# put all executables under build/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 1) Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2) Allow out‐of‐source builds:
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(WARNING "You are building in the source directory. \
We recommend creating a separate 'build/' folder.")
endif()

# 3) Find dependencies
find_package(OpenMP REQUIRED)
# if you later add JSON/YAML or HEALPix, find or add those here.

# 4) Specify include directories
#    - "include/" for your own headers
#    - "external/" for header‐only libraries
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/external
)

# 5) Gather your source files
set(SRCS
  src/PairHaloSimulator.cpp
  src/main_population.cpp
  src/helper.cpp
)

# 6) Declare the executable target
add_executable(halo_population ${SRCS})

# 7) Link with OpenMP
if (OpenMP_CXX_FOUND)
  target_link_libraries(halo_population PRIVATE OpenMP::OpenMP_CXX)
endif()

# 8) (Optional) Install rules
install(TARGETS halo_population RUNTIME DESTINATION bin)

# 9) (Optional) Tests, examples, etc.
# enable_testing()
# add_subdirectory(tests)
