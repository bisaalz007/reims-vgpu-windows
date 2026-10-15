# Keep the Windows resource compiler on LLVM with the C/C++ toolchain.
find_program(REIMS_RC llvm-windres REQUIRED)
set(CMAKE_RC_COMPILER "${REIMS_RC}" CACHE FILEPATH "Windows resource compiler" FORCE)
set(SDL_SHARED OFF CACHE BOOL "SDL is linked statically into the Reims DLL" FORCE)
