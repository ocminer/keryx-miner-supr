# Windows build of the in-process llama.cpp Vulkan engine.
#
# This is intentionally separate from tools/keryx-llama/CMakeLists.txt: the established NVIDIA
# release path configures that project with CUDA enabled, while this project is C++/Vulkan-only.
# Keeping the projects separate prevents an AMD workflow change from perturbing the working CUDA
# sidecar. The caller must append ../keryx-llama/keryx_vk_exports.inc.cpp to the pinned llama.cpp
# ggml-vulkan.cpp before configuring (the Linux build script uses the same source transformation).
cmake_minimum_required(VERSION 3.24)
project(keryx_llama_vk LANGUAGES C CXX)

if(NOT DEFINED LLAMA_SRC)
  message(FATAL_ERROR "Set -DLLAMA_SRC=<path to a clean llama.cpp b10015 checkout>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  # Mining rigs should not need a separately installed VC runtime. Apply /MT consistently to the
  # wrapper and every static llama/ggml library to avoid mixed-runtime allocation boundaries.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
endif()

# Static llama/ggml closure inside one DLL. The AMD driver supplies only the standard Vulkan
# loader/ICD at runtime; no loose llama or ggml DLL may escape into the release archive.
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(GGML_CUDA            OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN          ON  CACHE BOOL "" FORCE)
set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)

find_package(Vulkan COMPONENTS glslc REQUIRED)
add_subdirectory(${LLAMA_SRC} llama-build)

# The Keryx gather/walk shaders are separate from llama.cpp's generated Vulkan shaders. glslc's
# C output is an initializer list which keryx_llama_vk.cpp includes inside a uint32_t array.
set(KERYX_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/keryx-shaders")
file(MAKE_DIRECTORY "${KERYX_SHADER_DIR}")
set(KERYX_WALK_SPV  "${KERYX_SHADER_DIR}/pom_walk_vk.spv.inc")
set(KERYX_WALK_DOT_SPV "${KERYX_SHADER_DIR}/pom_walk_dot_vk.spv.inc")
set(KERYX_FETCH_SPV "${KERYX_SHADER_DIR}/pom_fetch_vk.spv.inc")
add_custom_command(
  OUTPUT "${KERYX_WALK_SPV}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -mfmt=c
          -o "${KERYX_WALK_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/../keryx-llama/pom_walk_vk.comp"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../keryx-llama/pom_walk_vk.comp"
  VERBATIM)
# Same v4 walk source with the packed int8 dot-product extension (selected at load time when ggml
# enabled VK_KHR_shader_integer_dot_product on the device).
add_custom_command(
  OUTPUT "${KERYX_WALK_DOT_SPV}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -DUSE_DOT=1 -mfmt=c
          -o "${KERYX_WALK_DOT_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/../keryx-llama/pom_walk_vk.comp"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../keryx-llama/pom_walk_vk.comp"
  VERBATIM)
add_custom_command(
  OUTPUT "${KERYX_FETCH_SPV}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -mfmt=c
          -o "${KERYX_FETCH_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/../keryx-llama/pom_fetch_vk.comp"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../keryx-llama/pom_fetch_vk.comp"
  VERBATIM)
# Two-phase variants (offset chase + pipelined walk, scalar and packed-dot) of the same source.
set(KERYX_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../keryx-llama/pom_walk_vk.comp")
set(KERYX_CHASE_SPV "${KERYX_SHADER_DIR}/pom_chase_vk.spv.inc")
set(KERYX_WALK_TP_SPV "${KERYX_SHADER_DIR}/pom_walk_tp_vk.spv.inc")
set(KERYX_WALK_TP_DOT_SPV "${KERYX_SHADER_DIR}/pom_walk_tp_dot_vk.spv.inc")
add_custom_command(OUTPUT "${KERYX_CHASE_SPV}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -DPOM_CHASE=1 -mfmt=c -o "${KERYX_CHASE_SPV}" "${KERYX_SHADER_SRC}"
  DEPENDS "${KERYX_SHADER_SRC}" VERBATIM)
add_custom_command(OUTPUT "${KERYX_WALK_TP_SPV}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -DPOM_TP=1 -mfmt=c -o "${KERYX_WALK_TP_SPV}" "${KERYX_SHADER_SRC}"
  DEPENDS "${KERYX_SHADER_SRC}" VERBATIM)
add_custom_command(OUTPUT "${KERYX_WALK_TP_DOT_SPV}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -DPOM_TP=1 -DUSE_DOT=1 -mfmt=c -o "${KERYX_WALK_TP_DOT_SPV}" "${KERYX_SHADER_SRC}"
  DEPENDS "${KERYX_SHADER_SRC}" VERBATIM)
set(KERYX_WALK_TP_CM_SPV "${KERYX_SHADER_DIR}/pom_walk_tp_cm_vk.spv.inc")
add_custom_command(OUTPUT "${KERYX_WALK_TP_CM_SPV}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O -DPOM_TP_CM=1 -mfmt=c -o "${KERYX_WALK_TP_CM_SPV}" "${KERYX_SHADER_SRC}"
  DEPENDS "${KERYX_SHADER_SRC}" VERBATIM)
add_custom_target(keryx-pom-vk-shaders DEPENDS "${KERYX_WALK_SPV}" "${KERYX_WALK_DOT_SPV}" "${KERYX_FETCH_SPV}"
  "${KERYX_CHASE_SPV}" "${KERYX_WALK_TP_SPV}" "${KERYX_WALK_TP_DOT_SPV}" "${KERYX_WALK_TP_CM_SPV}")

add_library(keryx-llama-vk SHARED "${CMAKE_CURRENT_SOURCE_DIR}/../keryx-llama/keryx_llama_vk.cpp")
add_dependencies(keryx-llama-vk keryx-pom-vk-shaders)
set_target_properties(keryx-llama-vk PROPERTIES PREFIX "" OUTPUT_NAME "keryx-llama-vk")
target_include_directories(keryx-llama-vk PRIVATE
  "${KERYX_SHADER_DIR}"
  "${LLAMA_SRC}/include"
  "${LLAMA_SRC}/ggml/include"
  "${LLAMA_SRC}/src"
  "${LLAMA_SRC}/common")
target_link_libraries(keryx-llama-vk PRIVATE llama ggml ggml-vulkan Vulkan::Vulkan)

# MSVC does not export extern-C symbols by default. Keep this explicit list in lockstep with the
# Rust loader and the picker functions it resolves before model allocation.
if(MSVC)
  set_target_properties(keryx-llama-vk PROPERTIES
    LINK_FLAGS "/DEF:\"${CMAKE_CURRENT_SOURCE_DIR}/keryx_llama_vk.def\"")
endif()
