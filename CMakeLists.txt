cmake_minimum_required(VERSION 3.20...3.31)
project(pylibresidfp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LIBRESIDFP_REPO "https://github.com/libsidplayfp/libresidfp.git"
    CACHE STRING "libresidfp git repository")
set(LIBRESIDFP_TAG "v1.2.3")
set(LIBRESIDFP_SHA "32c8525418a643eaed8dfafa3d7039e250050cfa")

find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(libresidfp
  GIT_REPOSITORY ${LIBRESIDFP_REPO}
  GIT_TAG ${LIBRESIDFP_SHA}
  PATCH_COMMAND sh -c "git apply -R --check \"$0\" 2>/dev/null || git apply \"$0\""
                ${CMAKE_CURRENT_SOURCE_DIR}/patches/0001-deterministic-rounding.patch)
FetchContent_MakeAvailable(libresidfp)
set(RES ${libresidfp_SOURCE_DIR})

include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
  "int main() { return __builtin_expect(0, 0); }" HAVE_BUILTIN_EXPECT_OK)
set(HAVE_BUILTIN_EXPECT 0)
if(HAVE_BUILTIN_EXPECT_OK)
  set(HAVE_BUILTIN_EXPECT 1)
endif()
set(RESIDFP_BRANCH_HINTS 1)
set(RESIDFP_INLINING 1)
set(RESIDFP_INLINE inline)
set(PACKAGE_VERSION "1.2.3")
set(LIB_MAJOR 1)
set(LIB_MINOR 2)
set(LIB_LEVEL 3)

set(GEN ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(${RES}/src/siddefs-fp.h.in ${GEN}/siddefs-fp.h @ONLY)
configure_file(${RES}/src/residfp/sidversion.h.in ${GEN}/residfp/sidversion.h
               @ONLY)

file(GLOB RES_SOURCES ${RES}/src/*.cpp ${RES}/src/resample/*.cpp)
list(FILTER RES_SOURCES EXCLUDE REGEX "resample/test\\.cpp$")
list(APPEND RES_SOURCES ${RES}/src/version.cc ${RES}/src/residfp/residfp.cpp)

pybind11_add_module(_core MODULE src/binding.cpp ${RES_SOURCES})
target_include_directories(_core PRIVATE ${GEN} ${RES}/src)
target_compile_options(_core PRIVATE -ffast-math -fno-unsafe-math-optimizations
                                     -ftree-loop-vectorize)
target_compile_definitions(_core PRIVATE VERSION_INFO="0.1.0")

install(TARGETS _core DESTINATION .)
