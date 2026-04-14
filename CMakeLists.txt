cmake_minimum_required(VERSION 3.22)
project(mlc_solver LANGUAGES CXX)

# ─────────────────────────────────────────────────────────────────────────────
#  C++ standard
# ─────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─────────────────────────────────────────────────────────────────────────────
#  Build type
#  Default to Release if not specified.  Override with:
#    cmake -DCMAKE_BUILD_TYPE=Debug ..
# ─────────────────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Build type: Debug | Release | RelWithDebInfo | MinSizeRel" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE defaulted to Release")
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  Compiler flags
# ─────────────────────────────────────────────────────────────────────────────
set(MLC_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# Release: maximise throughput on Google Compute-Optimised (Sapphire Rapids / Cascade Lake).
# Adjust -march if you know the exact microarchitecture; x86-64-v3 is a safe
# baseline for any GCP c2/c3 instance.
set(MLC_RELEASE_FLAGS
    -O3
    -march=x86-64-v3      # AVX2 + FMA baseline for c2/c3 instances
    -funroll-loops
    -fno-math-errno
    -DNDEBUG
)

# ─────────────────────────────────────────────────────────────────────────────
#  Locate HiGHS
#
#  Preferred install path (from source build or apt):
#    /usr/local/lib/cmake/highs/
#  Override with:
#    cmake -DHIGHS_DIR=/path/to/highs/lib/cmake/highs ..
# ─────────────────────────────────────────────────────────────────────────────
find_package(HIGHS REQUIRED
    HINTS
        /usr/local/lib/cmake/highs
        /usr/lib/cmake/highs
        $ENV{HIGHS_DIR}
    NO_DEFAULT_PATH
)

if(NOT TARGET highs::highs)
    # Fallback: some HiGHS builds export a lower-case target name.
    if(TARGET HiGHS::HiGHS)
        add_library(highs::highs ALIAS HiGHS::HiGHS)
    else()
        message(FATAL_ERROR
            "HiGHS target not found.  "
            "Set -DHIGHS_DIR=<install_prefix>/lib/cmake/highs or install via:\n"
            "  sudo apt install libhighs-dev     # Ubuntu 23.04+\n"
            "  # OR build from source:\n"
            "  git clone https://github.com/ERGO-Code/HiGHS && cd HiGHS\n"
            "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local\n"
            "  cmake --build build -j$(nproc) && sudo cmake --install build"
        )
    endif()
endif()

message(STATUS "Found HiGHS: ${HIGHS_VERSION}")
message(STATUS "HiGHS include dirs: ${HIGHS_INCLUDE_DIRS}")

# ─────────────────────────────────────────────────────────────────────────────
#  Executable target
# ─────────────────────────────────────────────────────────────────────────────
add_executable(mlc_solver main.cpp)

target_compile_options(mlc_solver PRIVATE
    ${MLC_WARNING_FLAGS}
    "$<$<CONFIG:Release>:${MLC_RELEASE_FLAGS}>"
    "$<$<CONFIG:RelWithDebInfo>:-O2;-g;-march=x86-64-v3>"
    "$<$<CONFIG:Debug>:-O0;-g3;-fsanitize=address,undefined>"
)

target_link_libraries(mlc_solver
    PRIVATE
        highs::highs
)

target_link_options(mlc_solver PRIVATE
    "$<$<CONFIG:Debug>:-fsanitize=address,undefined>"
)

# ─────────────────────────────────────────────────────────────────────────────
#  Install rule (optional)
# ─────────────────────────────────────────────────────────────────────────────
install(TARGETS mlc_solver
    RUNTIME DESTINATION bin
)

# ─────────────────────────────────────────────────────────────────────────────
#  Print configuration summary
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "══════════════════════════════════════════")
message(STATUS "  MLC Solver build configuration")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX        : ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID})")
message(STATUS "  Install to : ${CMAKE_INSTALL_PREFIX}/bin")
message(STATUS "══════════════════════════════════════════")
message(STATUS "")
