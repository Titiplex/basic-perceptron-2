cmake_minimum_required(VERSION 4.2)
project(TP1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Ne pas compiler les exemples et tests de la bibliothèque.
set(MATPLOTPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MATPLOTPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus.git
        GIT_TAG v1.2.2
        GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(matplotplusplus)

find_package(Threads REQUIRED)

function(download url)

    FetchContent_Declare(sonar URL ${url})
    FetchContent_MakeAvailable(sonar)
endfunction()

download(https://archive.ics.uci.edu/static/public/151/connectionist+bench+sonar+mines+vs+rocks.zip)

add_executable(TP1 main.cpp)

target_link_libraries(TP1
        PRIVATE
        matplot
        Threads::Threads
)