cmake_minimum_required(VERSION 3.27)
project(descriptorExtractor)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV, Boost, and TBB packages in the system
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(TBB REQUIRED)

message(STATUS "TBB_LIBRARIES: ${TBB_LIBRARIES}")


# Set compiler flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -march=native")

# Include directories for OpenCV, Boost, and TBB headers
#include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} /usr/include/oneapi/tbb)




# Define the executable
add_executable(descriptorExtractor
        descriptorExtractor/main.cpp
        descriptorExtractor/hpatchesDescriptorExtractor.cpp
        descriptorExtractor/descriptorProcessor.cpp
        descriptorExtractor/experimentConfig.hpp
        keypoints/VanillaSIFT.cpp
        keypoints/logger.hpp
        keypoints/logger.cpp
        #        keypoints/CHoNI.cpp
#        keypoints/CSIFT.cpp
#        keypoints/CSPIN.cpp
#        keypoints/DSPSIFT.cpp
#        keypoints/HoNC3.cpp
        #        keypoints/HoNI.cpp
#        keypoints/HoWH.cpp
#        keypoints/MSERSIFT.cpp
#        keypoints/OpponentSIFT.cpp
#        keypoints/RGBSIFT.cpp
#        keypoints/RGSIFT.cpp
#        keypoints/SPIN.cpp
)

# Link libraries to the executable
target_link_libraries(descriptorExtractor
        ${OpenCV_LIBS}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        tbb)
        #${TBB_LIBRARIES})
