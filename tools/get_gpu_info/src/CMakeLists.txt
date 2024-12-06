set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

add_executable(main main.cu)
