
#ifndef EIGENPY_EXPORT_H
#define EIGENPY_EXPORT_H

#ifdef EIGENPY_STATIC_DEFINE
#  define EIGENPY_EXPORT
#  define EIGENPY_NO_EXPORT
#else
#  ifndef EIGENPY_EXPORT
#    ifdef eigenpy_EXPORTS
        /* We are building this library */
#      define EIGENPY_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define EIGENPY_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef EIGENPY_NO_EXPORT
#    define EIGENPY_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef EIGENPY_DEPRECATED
#  define EIGENPY_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef EIGENPY_DEPRECATED_EXPORT
#  define EIGENPY_DEPRECATED_EXPORT EIGENPY_EXPORT EIGENPY_DEPRECATED
#endif

#ifndef EIGENPY_DEPRECATED_NO_EXPORT
#  define EIGENPY_DEPRECATED_NO_EXPORT EIGENPY_NO_EXPORT EIGENPY_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define EIGENPY_NO_DEPRECATED
#endif

#endif
