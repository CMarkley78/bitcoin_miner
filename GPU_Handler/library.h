#if defined(_MSC_VER)
    #define EXPORT __declspec(dllexport) //Shared lib shit for windows
#elif defined(__GNUC__)
    #define EXPORT __attribute__((visibility("default"))) //Shared lib shit for Linux
#else
    #define EXPORT //If we can't figure out where we are, just hope for the best with nothing
#endif

extern "C" {EXPORT void search (unsigned char * header_info, unsigned char* output);}
