//basicradial_lib_export.h

#ifndef EXAMPLE_EXPORT_H_
#define EXAMPLE_EXPORT_H_

#if defined (WIN32)
#if defined (basicradial_lib_EXPORTS)
#define EXPORTGADGETSEXAMPLE __declspec(dllexport)
#else
#define EXPORTGADGETSEXAMPLE __declspec(dllimport)
#endif
#else
#define EXPORTGADGETSEXAMPLE
#endif

#endif /* BASICRADIAL_LIB_EXPORT_H_ */

