
#ifndef EXAMPLE_EXPORT_H_
#define EXAMPLE_EXPORT_H_

#if defined (WIN32)
#if defined (radial_grappa_combined_lib_EXPORTS)
#define EXPORTGADGETSEXAMPLE __declspec(dllexport)
#else
#define EXPORTGADGETSEXAMPLE __declspec(dllimport)
#endif
#else
#define EXPORTGADGETSEXAMPLE
#endif

#endif

