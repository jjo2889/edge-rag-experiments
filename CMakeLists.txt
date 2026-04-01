cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
project(rag_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ONNXRUNTIME_ROOT_DEFAULT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.24.3")
set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT_DEFAULT}" CACHE PATH "Path to ONNX Runtime prefix")
if(DEFINED ENV{ONNXRUNTIME_ROOT} AND NOT "$ENV{ONNXRUNTIME_ROOT}" STREQUAL "")
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Path to ONNX Runtime prefix" FORCE)
endif()

set(LLAMA_CPP_ROOT_DEFAULT "${CMAKE_SOURCE_DIR}/third_party/llama.cpp")
set(LLAMA_CPP_ROOT "${LLAMA_CPP_ROOT_DEFAULT}" CACHE PATH "Path to llama.cpp root")
if(DEFINED ENV{LLAMA_CPP_ROOT} AND NOT "$ENV{LLAMA_CPP_ROOT}" STREQUAL "")
  set(LLAMA_CPP_ROOT "$ENV{LLAMA_CPP_ROOT}" CACHE PATH "Path to llama.cpp root" FORCE)
endif()

set(FAISS_ROOT "" CACHE PATH "Optional FAISS install prefix")
if(DEFINED ENV{FAISS_ROOT} AND NOT "$ENV{FAISS_ROOT}" STREQUAL "")
  set(FAISS_ROOT "$ENV{FAISS_ROOT}" CACHE PATH "Optional FAISS install prefix" FORCE)
endif()
if(FAISS_ROOT)
  list(PREPEND CMAKE_PREFIX_PATH
    "${FAISS_ROOT}"
    "${FAISS_ROOT}/lib/cmake/faiss"
    "${FAISS_ROOT}/share/faiss")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${ONNXRUNTIME_ROOT}")

find_package(OpenMP REQUIRED)
find_package(faiss CONFIG REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(onnxruntime CONFIG REQUIRED
  HINTS "${ONNXRUNTIME_ROOT}")

if(TARGET yaml-cpp::yaml-cpp)
  set(YAMLCPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(YAMLCPP_TARGET yaml-cpp)
endif()

find_library(LLAMA_LIBRARY
  NAMES llama
  PATHS
    "${LLAMA_CPP_ROOT}/build/bin"
    "${LLAMA_CPP_ROOT}/build/src"
    "${LLAMA_CPP_ROOT}/build"
  NO_DEFAULT_PATH
  REQUIRED)

set(RAG_CORE_SOURCES
  src/config.cpp
  src/document_loader.cpp
  src/text_chunker.cpp
  src/embedding_model.cpp
  src/faiss_index.cpp
  src/retriever.cpp
  src/prompt_builder.cpp
  src/generator.cpp
  src/rag_pipeline.cpp)

add_library(rag_core ${RAG_CORE_SOURCES})

target_include_directories(rag_core
  PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${ONNXRUNTIME_ROOT}/include"
    "${LLAMA_CPP_ROOT}/include"
    "${LLAMA_CPP_ROOT}/ggml/include"
    "${LLAMA_CPP_ROOT}/vendor")

target_compile_options(rag_core PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(rag_core
  PUBLIC
    onnxruntime::onnxruntime
    faiss
    OpenMP::OpenMP_CXX
    ${YAMLCPP_TARGET}
    ${LLAMA_LIBRARY})

set(RAG_LOCAL_RPATH
  "${LLAMA_CPP_ROOT}/build/bin;${LLAMA_CPP_ROOT}/build/src;${ONNXRUNTIME_ROOT}/lib;${ONNXRUNTIME_ROOT}/lib64")

add_executable(rag_ingest apps/ingest_main.cpp)
target_link_libraries(rag_ingest PRIVATE rag_core)
set_target_properties(rag_ingest PROPERTIES BUILD_RPATH "${RAG_LOCAL_RPATH}")

add_executable(rag_query apps/query_main.cpp)
target_link_libraries(rag_query PRIVATE rag_core)
set_target_properties(rag_query PROPERTIES BUILD_RPATH "${RAG_LOCAL_RPATH}")
