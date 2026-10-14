cmake_minimum_required(VERSION 4.2)

# vcpkg manifest mode: auto-activate from VCPKG_ROOT if no toolchain file was passed explicitly.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

project(ROPE VERSION 0.4.0 LANGUAGES CXX C)

# CI version override: -DROPE_RELEASE_VERSION=x.y.z (not ROPE_VERSION, which project() hardcodes).
string(REGEX REPLACE "^v" "" ROPE_RELEASE_VERSION "${ROPE_RELEASE_VERSION}")
string(STRIP "${ROPE_RELEASE_VERSION}" ROPE_RELEASE_VERSION)
if(NOT ROPE_RELEASE_VERSION STREQUAL "")
    if(NOT ROPE_RELEASE_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
        message(FATAL_ERROR "ROPE_RELEASE_VERSION must be X.Y.Z or vX.Y.Z (got '${ROPE_RELEASE_VERSION}')")
    endif()
    set(PROJECT_VERSION "${ROPE_RELEASE_VERSION}")
    string(REPLACE "." ";" _ver_parts "${ROPE_RELEASE_VERSION}")
    list(GET _ver_parts 0 PROJECT_VERSION_MAJOR)
    list(GET _ver_parts 1 PROJECT_VERSION_MINOR)
    list(GET _ver_parts 2 PROJECT_VERSION_PATCH)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Version header — generated into build tree
# ---------------------------------------------------------------------------
configure_file(cmake/version.h.in
    "${CMAKE_CURRENT_BINARY_DIR}/include/rope/core/version.h" @ONLY)

# ---------------------------------------------------------------------------
# Build type
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ROPE_USE_LIBTORCH  "Build the TorchScript / LibTorch backend" ON)
option(ROPE_BUILD_TESTS   "Build and register unit tests (CTest)"    OFF)
option(ROPE_USE_ZARR      "Build 'rope forecast --zarr' export support (needs zlib)" ON)

# ---------------------------------------------------------------------------
# External dependencies (nlohmann/json, CLI11) via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    URL      https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
    CLI11
    URL      https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.tar.gz
    URL_HASH SHA256=f2d893a65c3b1324c50d4e682c0cdc021dd0477ae2c048544f39eed6654b699a
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
    rope_registry
    URL      https://github.com/AssistLabOrg/rope-registry/archive/refs/tags/v0.5.tar.gz
    URL_HASH SHA256=e966efe87f34f77f1a3b59f1b0a6658bf14a8e9609e12511a23fe0cfe0774a59
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json CLI11 rope_registry)

include(cmake/RopeHelpers.cmake)

# ---------------------------------------------------------------------------
# ONNX Runtime + LibTorch (cmake/Dependencies.cmake; ROPE_DOWNLOAD_DEPS=ON to auto-fetch)
# ---------------------------------------------------------------------------
include(cmake/Dependencies.cmake)

if(ORT_INC AND ORT_LIB)
    message(STATUS "ONNX Runtime include: ${ORT_INC}")
    message(STATUS "ONNX Runtime library: ${ORT_LIB}")
else()
    message(STATUS "ONNX Runtime not found — forecast/ will not build")
endif()

# LibTorch — optional, needed only by forecast/
if(ROPE_USE_LIBTORCH)
    set(LIBTORCH_ROOT "" CACHE PATH
        "Root of LibTorch distribution (sets Torch_DIR if Torch_DIR not already set)")
    if(LIBTORCH_ROOT AND NOT Torch_DIR)
        set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch" CACHE PATH "" FORCE)
    endif()
    find_package(Torch QUIET)
    if(Torch_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
        message(STATUS "LibTorch found: ${TORCH_INCLUDE_DIRS}")
    else()
        message(STATUS "LibTorch not found — COAE TorchScript backend disabled")
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP — optional, used by forecast/ for parallel base-model rollouts
# ---------------------------------------------------------------------------
find_package(OpenMP)

# ---------------------------------------------------------------------------
# zlib — ForecastZarrWriter's chunk compressor (io/forecast_zarr_writer.cpp)
# ---------------------------------------------------------------------------
if(ROPE_USE_ZARR)
    find_package(ZLIB QUIET)
    if(NOT ZLIB_FOUND)
        message(STATUS "zlib not found — ROPE_USE_ZARR requested but disabled")
        set(ROPE_USE_ZARR OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# cpp-httplib + OpenSSL (vcpkg manifest mode) — DriverCacheManager's HTTPS client
# ---------------------------------------------------------------------------
find_package(httplib CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Compiler warnings and release optimisations
# ---------------------------------------------------------------------------
add_library(rope_compile_opts INTERFACE)
target_compile_options(rope_compile_opts INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3 -march=native>
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2>
)

# ---------------------------------------------------------------------------
# Symbol visibility — hidden by default on Unix shared libraries
# ---------------------------------------------------------------------------
set(CMAKE_C_VISIBILITY_PRESET   hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# All objects that end up in a shared library must be PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# rope_core — platform layer (paths, memory-mapped files), datetime, version
# ---------------------------------------------------------------------------
add_library(rope_core STATIC)

target_sources(rope_core PRIVATE
    $<IF:$<PLATFORM_ID:Windows>,
        src/core/platform/windows.cpp,
        src/core/platform/posix.cpp>
    src/core/version.cpp
)

target_include_directories(rope_core
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_link_libraries(rope_core
    PUBLIC  rope_compile_opts
)

# ---------------------------------------------------------------------------
# rope_io — CSV parsing, config, stats, driver DB, IC table
# ---------------------------------------------------------------------------
add_library(rope_io STATIC
    src/io/csv_reader.cpp
    src/io/config_reader.cpp
    src/io/driver_db.cpp
    src/io/driver_bin.cpp
    src/io/driver_cache.cpp
    src/io/ic_table.cpp
    src/io/ic_bin.cpp
    src/io/da_states.cpp
    src/io/model_manifest.cpp
    src/io/forecast_grid_bin.cpp
    src/io/mapped_forecast_grid.cpp
)
target_link_libraries(rope_io
    PUBLIC  rope_core
    PRIVATE nlohmann_json::nlohmann_json
)
if(ROPE_USE_ZARR)
    target_sources(rope_io PRIVATE src/io/forecast_zarr_writer.cpp)
    target_link_libraries(rope_io PRIVATE ZLIB::ZLIB)
    target_compile_definitions(rope_io PUBLIC ROPE_HAS_ZARR)
    message(STATUS "Zarr forecast export ('rope forecast --zarr'): enabled")
else()
    message(STATUS "Zarr forecast export ('rope forecast --zarr'): disabled")
endif()

# ---------------------------------------------------------------------------
# rope_net — HTTPS client (rope::net::IHttpClient); kept out of rope_io so capi stays free of TLS/OpenSSL.
# ---------------------------------------------------------------------------
include(cmake/EmbedCacert.cmake)

add_library(rope_net STATIC
    src/net/http_client.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/generated/cacert_pem.cpp"
)
target_include_directories(rope_net PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/net>
)
target_link_libraries(rope_net
    PUBLIC  rope_core
    PRIVATE httplib::httplib
)

# ---------------------------------------------------------------------------
# rope_interpolate — in-memory grid interpolation
# ---------------------------------------------------------------------------
add_library(rope_interpolate STATIC
    src/interpolate/grid_interpolator.cpp
    src/interpolate/config.cpp
)
target_link_libraries(rope_interpolate PUBLIC rope_core rope_io)

# ---------------------------------------------------------------------------
# rope_forecast — full inference pipeline
# ---------------------------------------------------------------------------
if(ORT_INC AND ORT_LIB)
    add_library(rope_forecast STATIC
        src/forecast/pipeline.cpp
        src/forecast/pipeline_registry.cpp
        src/forecast/config_builder.cpp
        src/forecast/backends/runtime_compat.cpp
        src/forecast/backends/model_factory.cpp
        src/forecast/backends/ic_source_factory.cpp
        src/forecast/backends/decoder_factory.cpp
        src/forecast/backends/coae_decoder.cpp
        src/forecast/stacked_ensemble/stacked_ensemble_pipeline.cpp
        src/forecast/stacked_ensemble/unscented_transform.cpp
        src/forecast/addons/f10_ensemble.cpp
    )
    target_include_directories(rope_forecast
        PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/forecast>
                $<BUILD_INTERFACE:${ORT_INC}>
    )
    target_link_libraries(rope_forecast
        PUBLIC  rope_core
                rope_io
        PRIVATE ${ORT_LIB}
                rope_net
                nlohmann_json::nlohmann_json
    )
    if(Torch_FOUND)
        target_link_libraries(rope_forecast PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(rope_forecast PRIVATE ${TORCH_INCLUDE_DIRS})
        target_compile_definitions(rope_forecast PRIVATE ROPE_USE_LIBTORCH)
    endif()
    if(ROPE_HARDWARE MATCHES "^cuda")
        # ONNX Runtime CUDA execution provider for the decoder (backends/onnx_model.h)
        target_compile_definitions(rope_forecast PRIVATE ROPE_ORT_CUDA)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(rope_forecast PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(rope_forecast PRIVATE ROPE_USE_OPENMP)
    endif()
    message(STATUS "rope_forecast: enabled")
else()
    message(STATUS "rope_forecast: disabled (ONNX Runtime not found)")
endif()

# ---------------------------------------------------------------------------
# rope (shared library) — C API
# ---------------------------------------------------------------------------
add_library(rope SHARED
    src/capi/rope.cpp
)
# Force "lib" prefix on all platforms (avoids Windows rope.dll colliding with the Rope.dll NuGet wrapper).
set_target_properties(rope PROPERTIES PREFIX "lib")
target_compile_definitions(rope PRIVATE ROPE_BUILDING_SHARED)
target_link_libraries(rope
    PRIVATE rope_io
            rope_interpolate
            nlohmann_json::nlohmann_json
)
# Runtime library path so the shared lib finds siblings in lib/
if(UNIX AND NOT APPLE)
    set_target_properties(rope PROPERTIES INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    set_target_properties(rope PROPERTIES INSTALL_RPATH "@loader_path")
endif()

# ---------------------------------------------------------------------------
# rope (executable) — CLI
# ---------------------------------------------------------------------------
add_executable(rope_exe
    src/cli/main.cpp
)
set_target_properties(rope_exe PROPERTIES OUTPUT_NAME rope)
target_link_libraries(rope_exe
    PRIVATE rope_io
            rope_interpolate
            CLI11::CLI11
            nlohmann_json::nlohmann_json
)
if(TARGET rope_forecast)
    target_link_libraries(rope_exe PRIVATE rope_forecast)
    target_compile_definitions(rope_exe PRIVATE ROPE_HAS_FORECAST)
endif()
rope_set_runtime_rpath(rope_exe)
# Supports both the tarball layout (bin/ + ../lib/) and the flat NuGet layout.
if(APPLE)
    set(_exe_install_rpath "@executable_path;@executable_path/../lib;@loader_path;@loader_path/../lib")
else()
    set(_exe_install_rpath "$ORIGIN;$ORIGIN/../lib")
endif()
set_target_properties(rope_exe PROPERTIES INSTALL_RPATH "${_exe_install_rpath}")

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
              "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
    DESTINATION .)

install(TARGETS rope_exe
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS rope
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}    # .dll on Windows
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})   # .lib import library on Windows

install(FILES include/rope/capi/rope.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rope/capi)
# Generate rope.conf with the correct decoder device for this hardware variant.
if(ROPE_HARDWARE MATCHES "^cuda")
    set(ROPE_DECODER_DEVICE "cuda")
elseif(ROPE_HARDWARE MATCHES "^rocm")
    set(ROPE_DECODER_DEVICE "cuda")
else()
    set(ROPE_DECODER_DEVICE "cpu")
endif()
configure_file(config/rope.conf.in "${CMAKE_CURRENT_BINARY_DIR}/config/rope.conf" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/config/rope.conf" DESTINATION config)

# Bundle ORT runtime libraries
if(ORT_LIB)
    get_filename_component(_ort_lib_dir "${ORT_LIB}" DIRECTORY)
    install(DIRECTORY "${_ort_lib_dir}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING
            PATTERN "libonnxruntime*.so*"
            PATTERN "libonnxruntime*.dylib"
            PATTERN "libonnxruntime_providers*.so*")
    if(WIN32)
        install(DIRECTORY "${_ort_lib_dir}/"
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            FILES_MATCHING PATTERN "onnxruntime*.dll")
    endif()
endif()

# Bundle LibTorch runtime libraries
if(Torch_FOUND)
    if(TORCH_INSTALL_PREFIX)
        set(_torch_lib_dir "${TORCH_INSTALL_PREFIX}/lib")
    else()
        list(GET TORCH_LIBRARIES 0 _first_torch_lib)
        get_filename_component(_torch_lib_dir "${_first_torch_lib}" DIRECTORY)
    endif()
    if(WIN32)
        install(DIRECTORY "${_torch_lib_dir}/"
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            FILES_MATCHING PATTERN "*.dll")
    elseif(APPLE)
        install(DIRECTORY "${_torch_lib_dir}/"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            FILES_MATCHING PATTERN "*.dylib")
    else()
        install(DIRECTORY "${_torch_lib_dir}/"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            FILES_MATCHING PATTERN "*.so" PATTERN "*.so.*")
    endif()
endif()

# Model artifacts
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/models/tiegcm-aurora-v1")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/models/tiegcm-aurora-v1/" DESTINATION data/models/tiegcm-aurora-v1)
endif()

# Optional override — point -DROPE_MODELS_DIR= at a different exported dir
set(ROPE_MODELS_DIR "" CACHE PATH
    "Directory of exported model artifacts to bundle in the package (overrides data/models/tiegcm-aurora-v1)")
if(ROPE_MODELS_DIR AND EXISTS "${ROPE_MODELS_DIR}")
    install(DIRECTORY "${ROPE_MODELS_DIR}/" DESTINATION data/models/tiegcm-aurora-v1)
endif()

# Driver (space-weather) data is not bundled — downloaded separately (README.md) or fetched by DriverCacheManager.

# Python binding
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/" DESTINATION python)

# .NET binding
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dotnet/" DESTINATION dotnet
    PATTERN "bin"    EXCLUDE
    PATTERN "obj"    EXCLUDE)

# Python demo
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo/pyproject.toml")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/demo/" DESTINATION demo
    PATTERN "__pycache__"     EXCLUDE
    PATTERN "*.pyc"           EXCLUDE
    PATTERN "*.pkl"           EXCLUDE
    PATTERN "*.egg-info"      EXCLUDE
    PATTERN "output"          EXCLUDE)
endif()

# ---------------------------------------------------------------------------
# CPack
# ---------------------------------------------------------------------------
set(CPACK_PACKAGE_NAME             "rope_framework")
set(CPACK_PACKAGE_VENDOR           "ROPE")
set(CPACK_PACKAGE_VERSION          "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "ROPE atmospheric density forecasting runtime")
set(CPACK_STRIP_FILES TRUE)

if(NOT ROPE_PACKAGE_SUFFIX)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(_cpack_os "macos")
    else()
        string(TOLOWER "${CMAKE_SYSTEM_NAME}" _cpack_os)
    endif()
    set(ROPE_PACKAGE_SUFFIX "${_cpack_os}-${CMAKE_SYSTEM_PROCESSOR}-${ROPE_HARDWARE}")
endif()
set(CPACK_PACKAGE_FILE_NAME
    "rope_framework-${PROJECT_VERSION}-${ROPE_PACKAGE_SUFFIX}")

if(WIN32)
    set(CPACK_GENERATOR "ZIP")
else()
    set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)

# ---------------------------------------------------------------------------
# Unit tests (opt-in with -DROPE_BUILD_TESTS=ON)
# ---------------------------------------------------------------------------
if(ROPE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
