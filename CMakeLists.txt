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

set(UPSTREAM ${CMAKE_CURRENT_SOURCE_DIR}/extern/libresidfp)
if(NOT EXISTS ${UPSTREAM}/src/SID.cpp)
  message(FATAL_ERROR "extern/libresidfp is empty: git submodule update --init")
endif()

find_package(pybind11 CONFIG REQUIRED)

find_program(PATCH_EXE patch REQUIRED)
set(RES ${CMAKE_CURRENT_BINARY_DIR}/libresidfp)
file(REMOVE_RECURSE ${RES})
file(COPY ${UPSTREAM}/src DESTINATION ${RES})
file(GLOB PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch)
foreach(p ${PATCHES})
  execute_process(COMMAND ${PATCH_EXE} -p1 -i ${p} WORKING_DIRECTORY ${RES}
                  RESULT_VARIABLE rc)
  if(rc)
    message(FATAL_ERROR "patch ${p} failed")
  endif()
endforeach()

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
