cmake_minimum_required(VERSION 2.8)

project(basicradial)

# File modified from CMake files for through time Cartesian grappa and radial gadgets

if (WIN32)
ADD_DEFINITIONS(-DWIN32 -D_WIN32 -D_WINDOWS)
ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
endif (WIN32)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

###############################################################
#Bootstrap search for libraries 
# (We need to find cmake modules in Gadgetron)
###############################################################
find_path(GADGETRON_CMAKE_MODULES FindGadgetron.cmake HINTS
$ENV{GADGETRON_HOME}/cmake
/usr/local/gadgetron
/usr/local/share/gadgetron/cmake)

if (NOT GADGETRON_CMAKE_MODULES)
  MESSAGE(FATAL_ERROR "GADGETRON_CMAKE_MODULES cannot be found. 
   Try to set GADGETRON_HOME environment variable.")
endif(NOT GADGETRON_CMAKE_MODULES)

set(CMAKE_MODULE_PATH ${GADGETRON_CMAKE_MODULES})
###############################################################

find_package(ISMRMRD REQUIRED)
find_package(Gadgetron REQUIRED)
find_package(Boost REQUIRED)
find_package(ACE REQUIRED)
find_package(CUDA)

if (CUDA_FOUND)
  ADD_DEFINITIONS(-DUSE_CUDA)
  include_directories( ${CUDA_INCLUDE_DIRS} )
  if (NOT WIN32)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  endif (NOT WIN32)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} )
  set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  MESSAGE("Compiling with ${CUDA_NVCC_FLAGS}")
else (CUDA_FOUND)
  MESSAGE("CUDA not found. CUDA components will not be compiled.")
  SET( CUDA_NVCC_FLAGS "NA")
endif (CUDA_FOUND)

set(CMAKE_INSTALL_PREFIX ${GADGETRON_HOME})

INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}/gadgets/mri_core
    ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
    ${CMAKE_SOURCE_DIR}/toolboxes/core
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/image
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
    ${CMAKE_SOURCE_DIR}/toolboxes/core/gpu
    ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/fft/gpu
    ${CMAKE_SOURCE_DIR}/toolboxes/mri_core
#Added from CMakeList for radial gadgets
    ${CMAKE_SOURCE_DIR}/toolboxes/nfft/gpu
    ${CMAKE_SOURCE_DIR}/toolboxes/solvers
    ${CMAKE_SOURCE_DIR}/toolboxes/solvers/gpu
    ${CMAKE_SOURCE_DIR}/toolboxes/operators
    ${CMAKE_SOURCE_DIR}/toolboxes/operators/gpu
    ${ARMADILLO_INCLUDE_DIRS}
    ${ACE_INCLUDE_DIR} 
    ${Boost_INCLUDE_DIR}
    ${GADGETRON_INCLUDE_DIR}
    ${ISMRMRD_INCLUDE_DIR}
    ${ISMRMRD_SCHEMA_DIR}
    ${ISMRMRD_XSD_INCLUDE_DIR}
     )

LINK_DIRECTORIES(${GADGETRON_LIB_DIR})

ADD_LIBRARY(basicradial_lib SHARED 
NFFT2DGadget_basic.h
NFFT2DGadget_basic.cpp
PassThroughGadget.cpp
PassThroughGadget.h
)

TARGET_LINK_LIBRARIES(basicradial_lib
      gadgetron_grappa
      gadgetron_gadgetbase
      gadgetron_toolbox_log
      gadgetron_toolbox_gpuparallelmri
      gadgetron_toolbox_cpucore
      gadgetron_toolbox_cpucore_math
      gadgetron_toolbox_cpufft
      gadgetron_toolbox_gpufft
      gadgetron_toolbox_mri_core
      gadgetron_mricore
#Added from CMake file for radial gadgets
	gadgetron_gpuradial
	gadgetron_toolbox_gpunfft
	gadgetron_toolbox_gpusolvers
	gadgetron_toolbox_gpuoperators
	gadgetron_toolbox_gpucore
      ${Boost_LIBRARIES}
      ${ISMRMRD_LIBRARIES} ${FFTW3_LIBRARIES}
	${CUDA_LIBRARIES}
      optimized ${ACE_LIBRARIES} 
      debug ${ACE_DEBUG_LIBRARY}
)

INSTALL (FILES
         basicradial_lib_export.h
	NFFT2DGadget_basic.h
	PassThroughGadget.h
         DESTINATION include)

INSTALL(TARGETS basicradial_lib DESTINATION lib)

INSTALL(FILES basic_radial_gridding.xml DESTINATION share/gadgetron/config)
