# Cross-platform build of the in-process llama.cpp engine as a shared library
# (keryx-llama.dll on Windows / libkeryx-llama.so elsewhere). Used by the Windows CI
# (windows.yml) to produce keryx-llama.dll; the Linux/macOS packages keep building the
# .so via hiveos/build-keryx-llama.sh (unchanged, byte-identical), so this file is the
# Windows path only for now.
#
# Configure with the llama.cpp checkout dir:
#   cmake -S tools/keryx-llama -B build -DLLAMA_SRC=<path-to-llama.cpp> \
#         -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;90-virtual"
#   cmake --build build --config Release --target keryx-llama
# Output: <build>/keryx-llama.dll  (MSVC: <build>/Release/keryx-llama.dll)
#
# The 90-virtual entry emits compute_90 PTX that the driver forward-JITs to sm_120 (Blackwell:
# 5070Ti/5080/5090) at load — the same forward-compat trick the candle PoM exe uses (sm_70 PTX),
# so a CUDA 12.5 toolchain that predates sm_120 SASS still runs on 50-series cards.
cmake_minimum_required(VERSION 3.24)
project(keryx_llama LANGUAGES CXX CUDA)

if(NOT DEFINED LLAMA_SRC)
  message(FATAL_ERROR "Set -DLLAMA_SRC=<path to a llama.cpp checkout (pinned tag b10015)>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build llama.cpp as static libs with the CUDA backend, no curl, no host-arch autodetect
# (GGML_NATIVE OFF keeps the SIMD baseline predictable, matching the Linux .so).
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL          OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER  OFF CACHE BOOL "" FORCE)
set(GGML_CUDA           ON  CACHE BOOL "" FORCE)
set(GGML_NATIVE         OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_NCCL      OFF CACHE BOOL "" FORCE)

# H15 canonical trace: keryx_llama.cpp links against the q8canon leaves added to ggml-cuda by
# patches/*.patch. Apply them to the (pristine, pinned b10015) checkout at configure time; a tree
# that already carries them is accepted, anything else is a hard error (never build an engine
# that cannot commit a block-task root).
find_package(Git REQUIRED)
file(GLOB _keryx_patches "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
list(SORT _keryx_patches)
foreach(_p IN LISTS _keryx_patches)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_SRC}" apply --ignore-whitespace --check "${_p}"
                  RESULT_VARIABLE _fresh OUTPUT_QUIET ERROR_QUIET)
  if(_fresh EQUAL 0)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_SRC}" apply --ignore-whitespace "${_p}" RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "keryx-llama: failed to apply ${_p} to ${LLAMA_SRC}")
    endif()
    message(STATUS "keryx-llama: applied ${_p}")
  else()
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_SRC}" apply --ignore-whitespace --reverse --check "${_p}"
                    RESULT_VARIABLE _applied OUTPUT_QUIET ERROR_QUIET)
    if(NOT _applied EQUAL 0)
      message(FATAL_ERROR "keryx-llama: ${LLAMA_SRC} neither accepts nor already carries ${_p} (wrong llama.cpp pin?)")
    endif()
    message(STATUS "keryx-llama: ${_p} already applied")
  endif()
endforeach()
if(NOT EXISTS "${LLAMA_SRC}/ggml/src/ggml-cuda/q8canon.cu")
  message(FATAL_ERROR "keryx-llama: canonical-trace patch not present in ${LLAMA_SRC}")
endif()
if(NOT EXISTS "${LLAMA_SRC}/ggml/src/ggml-cuda/q8seq.cu")
  message(FATAL_ERROR "keryx-llama: sequence-leaves patch (0002) not present in ${LLAMA_SRC}")
endif()

add_subdirectory(${LLAMA_SRC} llama-build)

add_library(keryx-llama SHARED keryx_llama.cpp)
set_target_properties(keryx-llama PROPERTIES PREFIX "" OUTPUT_NAME "keryx-llama")
target_include_directories(keryx-llama PRIVATE
  ${LLAMA_SRC}/include
  ${LLAMA_SRC}/ggml/include
  ${LLAMA_SRC}/src
  ${LLAMA_SRC}/common)
# llama pulls in ggml + its cuda/cpu/base backends transitively.
target_link_libraries(keryx-llama PRIVATE llama ggml)

# Architecture gate (v0.14.1): the engine refuses GPUs below the lowest compiled arch with an
# actionable message instead of ggml_abort-ing the miner on the first kernel launch. Derived from
# the first CMAKE_CUDA_ARCHITECTURES entry ("75-real" -> 75).
if(CMAKE_CUDA_ARCHITECTURES)
  list(GET CMAKE_CUDA_ARCHITECTURES 0 _keryx_min_arch)
  string(REGEX REPLACE "[^0-9].*$" "" _keryx_min_arch "${_keryx_min_arch}")
  if(_keryx_min_arch MATCHES "^[0-9]+$")
    target_compile_definitions(keryx-llama PRIVATE KERYX_LLAMA_MIN_CC=${_keryx_min_arch})
    message(STATUS "keryx-llama: KERYX_LLAMA_MIN_CC=${_keryx_min_arch}")
  endif()
endif()

# MSVC: nothing is exported from a DLL by default — feed the module-definition file so the
# keryx_llama_* C ABI is exported (Linux/macOS export via default visibility, so no-op there).
if(MSVC)
  set_target_properties(keryx-llama PROPERTIES
    LINK_FLAGS "/DEF:\"${CMAKE_CURRENT_SOURCE_DIR}/keryx_llama.def\"")
endif()
