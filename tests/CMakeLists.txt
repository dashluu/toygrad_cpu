# 'tests' is the subproject name
project(tests)

# 'googletest' is the folder with Google Test sources
add_subdirectory(googletest)
include_directories(googletest/googletest/include googletest/googletest)

# 'googletest_run' is the target name
add_executable(googletest_run tensor_test.cpp nn_test.cpp)
target_link_libraries(googletest_run toygrad_cpu_lib)
target_link_libraries(googletest_run gtest gtest_main)