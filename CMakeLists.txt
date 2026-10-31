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
set(CONFIGURE_AC ${UPSTREAM}/configure.ac)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CONFIGURE_AC})
foreach(part major minor level)
  file(STRINGS ${CONFIGURE_AC} line
       REGEX "^m4_define\\(\\[lib_${part}\\], \\[[0-9]+\\]\\)" LIMIT_COUNT 1)
  string(REGEX REPLACE "^.*, \\[([0-9]+)\\]\\)$" "\\1" value "${line}")
  if(NOT value MATCHES "^[0-9]+$")
    message(FATAL_ERROR "lib_${part} not found in ${CONFIGURE_AC}")
  endif()
  string(TOUPPER ${part} upper)
  set(LIB_${upper} ${value})
endforeach()
set(PACKAGE_VERSION "${LIB_MAJOR}.${LIB_MINOR}.${LIB_LEVEL}")

file(GLOB PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch)
if(PACKAGE_VERSION VERSION_LESS 1.2.3)
  file(GLOB OLD_PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/patches/before-1.2.3/*.patch)
  list(APPEND PATCHES ${OLD_PATCHES})
endif()
foreach(p ${PATCHES})
  execute_process(COMMAND ${PATCH_EXE} -p1 -i ${p} WORKING_DIRECTORY ${RES}
                  RESULT_VARIABLE rc)
  if(rc)
    message(FATAL_ERROR "patch ${p} failed")
  endif()
endforeach()

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
target_compile_definitions(_core PRIVATE VERSION_INFO="${SKBUILD_PROJECT_VERSION}")

install(TARGETS _core DESTINATION .)
