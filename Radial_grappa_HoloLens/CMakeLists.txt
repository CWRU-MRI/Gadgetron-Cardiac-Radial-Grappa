cmake_minimum_required(VERSION 2.8)

project(radial_grappa)

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

# build options for OpenMP support
 find_package(OpenMP)
 option(USE_OPENMP "Use OpenMP" On)
 if (OPENMP_FOUND)
  if(USE_OPENMP)
    message("OpenMP multithreading enabled")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   add_definitions(-DUSE_OMP)
  else ()
    message("OpenMP multithreading is supported, but disabled")
  endif()
else ()
  message("OpenMP multithreading not supported")
 endif ()
 
include_directories(".")
set(CMAKE_INSTALL_PREFIX ${GADGETRON_HOME})
add_subdirectory(radial_grappa_combined_gpu)

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

ADD_LIBRARY(radial_grappa_combined_lib SHARED 
RadialGrappaCombinedReconstructionGadget.cpp
RadialGrappaCombinedReconstructionGadget.h
RadialGrappaCombinedCalibrationGadget.cpp
RadialGrappaCombinedCalibrationGadget.h
RadialGrappaCombinedReferencePrepGadget.cpp
RadialGrappaCombinedReferencePrepGadget.h
GrappaCombinedCoilCompressionGadget_Reconstruction.cpp
GrappaCombinedCoilCompressionGadget_Reconstruction.h
GrappaCombinedCoilCompressionGadget_Calibration.cpp
GrappaCombinedCoilCompressionGadget_Calibration.h
TileSliceGadget.h
TileSliceGadget.cpp
ImageArraySplitDFGadget.h
ImageArraySplitDFGadget.cpp
BucketToBufferDFGadget.h
BucketToBufferDFGadget.cpp
ComplexToFloatDFGadget.h
ComplexToFloatDFGadget.cpp
FloatToFixPointDFGadget.h
FloatToFixPointDFGadget.cpp
RemoveROOversamplingDFGadget.h
RemoveROOversamplingDFGadget.cpp
)

TARGET_LINK_LIBRARIES(radial_grappa_combined_lib
	radial_grappa_combined_gpu_lib
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

TARGET_INCLUDE_DIRECTORIES(radial_grappa_combined_lib PUBLIC
 radial_grappa_combined_gpu
)

INSTALL (FILES
RadialGrappaCombinedCalibrationGadget.h
RadialGrappaCombinedReferencePrepGadget.h
RadialGrappaCombinedReconstructionGadget.h
GrappaCombinedCoilCompressionGadget_Calibration.h
GrappaCombinedCoilCompressionGadget_Reconstruction.h
TileSliceGadget.h
ImageArraySplitDFGadget.h
BucketToBufferDFGadget.h
ComplexToFloatDFGadget.h
FloatToFixPointDFGadget.h
RemoveROOversamplingDFGadget.h
         DESTINATION include)

INSTALL(TARGETS radial_grappa_combined_lib DESTINATION lib)

INSTALL(FILES radial_grappa_combined_calibration.xml radial_grappa_combined_reconstruction.xml radial_grappa_combined_reconstruction_timing.xml 
DESTINATION share/gadgetron/config)
