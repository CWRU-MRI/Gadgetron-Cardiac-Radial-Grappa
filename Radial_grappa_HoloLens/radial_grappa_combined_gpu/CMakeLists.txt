project(radial_grappa_combined_gpu)

if (WIN32)
  ADD_DEFINITIONS(-D__BUILD_GADGETRON_GPUPMRI__)
  ADD_DEFINITIONS(-DWIN32_LEAN_AND_MEAN)
endif (WIN32)

if(WIN32)
  link_directories(${Boost_LIBRARY_DIRS})
endif(WIN32)

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

#set(CMAKE_INSTALL_PREFIX ${GADGETRON_HOME})

#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../radial_grappa_combined_gpu)
#SET(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/radial_grappa_combined_gpu/RadialGrappaCombinedReconstruction_GPUfxns.h) 
#SET(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/radial_grappa_combined_gpu/RadialGrappaCombinedReconstruction_GPUfxns.cu${HEADER_FILES})

set(ttc_htgrappa_SOURCES RadialGrappaCombinedReconstruction_GPUfxns.cpp RadialGrappaCombinedReconstruction_GPUfxns.cu)
set(ttc_htgrappa_HEADERS RadialGrappaCombinedReconstruction_GPUfxns.h)

include_directories(  
  ${CMAKE_SOURCE_DIR}/gadgets/mri_core
  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/operators
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/solvers
  ${CMAKE_SOURCE_DIR}/toolboxes/solvers/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/image
  ${CMAKE_SOURCE_DIR}/toolboxes/mri_core
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
  ${CUDA_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS} 
  ${ACE_INCLUDE_DIR} 
  ${GADGETRON_INCLUDE_DIR}
  ${ISMRMRD_INCLUDE_DIR}
  ${ISMRMRD_SCHEMA_DIR}
  ${ISMRMRD_XSD_INCLUDE_DIR}
  )

set(HEADERS
   ${HEADERS}
   ${CMAKE_CURRENT_SOURCE_DIR}/RadialGrappaCombinedReconstruction_GPUfxns.h
   PARENT_SCOPE
)

cuda_add_library(radial_grappa_combined_gpu_lib SHARED 
    RadialGrappaCombinedReconstruction_GPUfxns.h
    RadialGrappaCombinedReconstruction_GPUfxns.cu
RadialGrappaCombinedReconstruction_GPUfxns.cpp
    radial_grappa_combined_gpu_lib_export.h
${radial_grappa_combined_gpu_SOURCES}
  )

LINK_DIRECTORIES(${GADGETRON_LIB_DIR})

target_link_libraries(radial_grappa_combined_gpu_lib 
  gadgetron_grappa
  gadgetron_gadgetbase
  gadgetron_mricore
  gadgetron_toolbox_log
  gadgetron_toolbox_gpuparallelmri
  gadgetron_toolbox_gpucore
  gadgetron_toolbox_log
  gadgetron_toolbox_gpunfft
  gadgetron_toolbox_gpufft 
  gadgetron_toolbox_cpufft
  gadgetron_toolbox_cpucore
  gadgetron_toolbox_cpucore_math
  gadgetron_toolbox_mri_core
  ${Boost_LIBRARIES}
  ${FFTW3_LIBRARIES} 
  ${CUDA_LIBRARIES} 
  ${CUDA_CUFFT_LIBRARIES} 
  ${CUDA_CUBLAS_LIBRARIES} 
  ${ISMRMRD_LIBRARIES}
      optimized ${ACE_LIBRARIES} 
      debug ${ACE_DEBUG_LIBRARY}
  )

install(FILES 
	RadialGrappaCombinedReconstruction_GPUfxns.h
	radial_grappa_combined_gpu_lib_export.h
DESTINATION include COMPONENT main)

install(TARGETS radial_grappa_combined_gpu_lib DESTINATION lib COMPONENT main)


