#ifndef radial_grappa_combined_GPU_LB_EXPORT_H_
#define radial_grappa_combined_GPU_LB_EXPORT_H_

#if defined (WIN32)
#if defined (radial_grappa_combined_gpu_lib_EXPORTS)
#define EXPORTGPUMRI __declspec(dllexport)
#else
#define EXPORTGPUMRI __declspec(dllimport)
#endif
#else
#define EXPORTGPUMRI
#endif

#endif 
