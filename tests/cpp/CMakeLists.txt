include(FetchContent)

FetchContent_Declare(
    Catch2
    URL      https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
    URL_HASH SHA256=c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

set(ROPE_FIXTURE_DIR        "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures")
set(ROPE_FIXTURE_MODELS     "${ROPE_FIXTURE_DIR}/test_models")
set(ROPE_FIXTURE_MODELS_M3  "${ROPE_FIXTURE_DIR}/test_models_m3")
set(ROPE_FIXTURE_MODELS_SPLIT "${ROPE_FIXTURE_DIR}/test_models_split")
set(ROPE_CPP_FIXTURE_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
set(ROPE_REGISTRY_DIR       "${rope_registry_SOURCE_DIR}")

add_executable(rope_tests
    test_interpolator.cpp
    test_csv_reader.cpp
    test_config_reader.cpp
    test_datetime.cpp
    test_version.cpp
    test_ic_table.cpp
    test_driver_db.cpp
    test_driver_cache.cpp
    test_model_manifest.cpp
    test_grid_stitch.cpp
    test_kind_registry.cpp
    test_ic_kind_registry.cpp
    test_decoder_kind_registry.cpp
    test_driver_registry.cpp
    test_manifest_schema_fixtures.cpp
    test_forecast_grid_bin.cpp
    test_mapped_forecast_grid.cpp
)
target_link_libraries(rope_tests
    PRIVATE Catch2::Catch2WithMain
            rope_interpolate
            rope_io
            rope_core
            nlohmann_json::nlohmann_json
)
if(ROPE_USE_ZARR)
    target_sources(rope_tests PRIVATE test_forecast_zarr_writer.cpp)
    target_link_libraries(rope_tests PRIVATE ZLIB::ZLIB)
endif()
target_include_directories(rope_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/forecast"
)
target_compile_definitions(rope_tests PRIVATE
    ROPE_FIXTURE_DIR="${ROPE_FIXTURE_DIR}"
    ROPE_FIXTURE_MODELS="${ROPE_FIXTURE_MODELS}"
    ROPE_CPP_FIXTURE_DIR="${ROPE_CPP_FIXTURE_DIR}"
    ROPE_REGISTRY_DIR="${ROPE_REGISTRY_DIR}"
)
catch_discover_tests(rope_tests)

# CLI-level check: `rope convert-sw` on a raw CelesTrak CSV; runs unconditionally, unlike the pipeline tests below.
add_test(
    NAME    rope_convert_sw_detects_raw_celestrak_format
    COMMAND rope_exe convert-sw
            --input  "${ROPE_FIXTURE_DIR}/sw_celestrak_test.csv"
            --output "${CMAKE_CURRENT_BINARY_DIR}/convert_sw_celestrak_test.swbin"
)
set_tests_properties(rope_convert_sw_detects_raw_celestrak_format PROPERTIES
    PASS_REGULAR_EXPRESSION "Detected raw CelesTrak format"
)

if(TARGET rope_forecast)
    add_executable(rope_pipeline_tests
        test_pipeline.cpp
        test_manifest_failures.cpp
        test_streaming_pipeline.cpp
        test_addons.cpp
    )
    target_link_libraries(rope_pipeline_tests
        PRIVATE Catch2::Catch2WithMain
                rope_forecast
                rope_io
                rope_core
                nlohmann_json::nlohmann_json
    )
    # test_addons.cpp exercises the F10 ensemble directly (src/forecast/addons/).
    target_include_directories(rope_pipeline_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/forecast"
    )
    target_compile_definitions(rope_pipeline_tests PRIVATE
        ROPE_FIXTURE_DIR="${ROPE_FIXTURE_DIR}"
        ROPE_FIXTURE_MODELS="${ROPE_FIXTURE_MODELS}"
        ROPE_FIXTURE_MODELS_M3="${ROPE_FIXTURE_MODELS_M3}"
        ROPE_FIXTURE_MODELS_SPLIT="${ROPE_FIXTURE_MODELS_SPLIT}"
        ROPE_CPP_FIXTURE_DIR="${ROPE_CPP_FIXTURE_DIR}"
    )
    rope_set_runtime_rpath(rope_pipeline_tests)

    if(Torch_FOUND)
        add_executable(gen_decoder_pt ../gen_decoder_pt.cpp)
        target_link_libraries(gen_decoder_pt PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(gen_decoder_pt PRIVATE ${TORCH_INCLUDE_DIRS})

        add_test(
            NAME    gen_coae_decoder_pt
            COMMAND gen_decoder_pt "${ROPE_FIXTURE_MODELS}/coae_decoder.pt"
        )
        set_tests_properties(gen_coae_decoder_pt PROPERTIES
            FIXTURES_SETUP rope_pipeline_fixtures
        )
    endif()

    if(Torch_FOUND)
        catch_discover_tests(rope_pipeline_tests
            PROPERTIES FIXTURES_REQUIRED rope_pipeline_fixtures
        )
    else()
        catch_discover_tests(rope_pipeline_tests)
    endif()

    # Only meaningful in a ROPE_USE_ZARR=OFF build; doesn't run otherwise.
    if(NOT ROPE_USE_ZARR)
        add_test(
            NAME    rope_zarr_off_fails_loudly
            COMMAND rope_exe forecast --start "2024-01-01 00:00:00" --horizon 1
                    --zarr "${CMAKE_CURRENT_BINARY_DIR}/zarr_off_test"
        )
        set_tests_properties(rope_zarr_off_fails_loudly PROPERTIES
            PASS_REGULAR_EXPRESSION "without Zarr support"
        )
    endif()
endif()

if(TARGET rope_forecast)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
        add_test(
            NAME    rope_cli_tests
            COMMAND ${Python3_EXECUTABLE} -m pytest
                    ${CMAKE_CURRENT_SOURCE_DIR}/../python/test_cli.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/../python/test_zarr_export.py
                    -v --tb=short
        )
        set_tests_properties(rope_cli_tests PROPERTIES
            ENVIRONMENT
                "ROPE_LIB=$<TARGET_FILE:rope>;ROPE_EXE=$<TARGET_FILE:rope_exe>;ROPE_FIXTURE_DIR=${ROPE_FIXTURE_DIR}"
        )
        if(Torch_FOUND)
            set_tests_properties(rope_cli_tests PROPERTIES
                FIXTURES_REQUIRED rope_pipeline_fixtures
            )
        endif()
    endif()
endif()
