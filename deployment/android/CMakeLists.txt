cmake_minimum_required(VERSION 3.22.1)
project("edgestereoinr_qnn")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android NDK log library
find_library(log-lib log)
find_library(android-lib android)

# Qualcomm QNN SDK includes (configured via QNN_SDK_ROOT environment variable)
if(DEFINED ENV{QNN_SDK_ROOT})
    include_directories($ENV{QNN_SDK_ROOT}/include/QNN)
    link_directories($ENV{QNN_SDK_ROOT}/lib/aarch64-android)
endif()

add_library(
    edgestereoinr_qnn
    SHARED
    qnn_native_bridge.cpp
)

target_link_libraries(
    edgestereoinr_qnn
    ${log-lib}
    ${android-lib}
)
