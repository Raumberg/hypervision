cdef extern from "windows.h":
    # Type definitions
    ctypedef unsigned long DWORD
    ctypedef long LONG
    ctypedef unsigned int UINT
    ctypedef void* ULONG_PTR

    # Constants
    cdef enum:
        INPUT_MOUSE = 0
        MOUSEEVENTF_MOVE = 0x0001
        MOUSEEVENTF_ABSOLUTE = 0x8000
    
    # Structures
    ctypedef struct MOUSEINPUT:
        LONG dx
        LONG dy
        DWORD mouseData
        DWORD dwFlags
        DWORD time
        ULONG_PTR dwExtraInfo

    ctypedef struct INPUT:
        DWORD type
        MOUSEINPUT mi

    # stdcall windows.h function
    UINT __stdcall SendInput(UINT cInputs, INPUT *pInputs, int cbSize)

# inlining coords casting
cdef inline DWORD _to_absolute(int pos, int screen_size) nogil:
    return <DWORD>((pos * 65535) // screen_size)

# Interface functions
cpdef void set_abs_position(int x, int y, int screen_width, int screen_height) noexcept:
    cdef INPUT inp
    inp.type = INPUT_MOUSE
    inp.mi.dx = _to_absolute(x, screen_width)
    inp.mi.dy = _to_absolute(y, screen_height)
    inp.mi.mouseData = 0
    inp.mi.dwFlags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE
    inp.mi.time = 0
    inp.mi.dwExtraInfo = <ULONG_PTR>0 # reinterpret_cast(ulong, nullptr)
    
    SendInput(1, &inp, sizeof(INPUT))

cpdef void set_rel_position(int x, int y, float sensitivity, int screen_width, int screen_height) noexcept:
    cdef int center_x = screen_width // 2
    cdef int center_y = screen_height // 2
    cdef float sens = sensitivity if sensitivity != 0.0 else 1.0
    
    cdef INPUT inp
    inp.type = INPUT_MOUSE
    inp.mi.dx = <LONG>((x - center_x) * sens)
    inp.mi.dy = <LONG>((y - center_y) * sens)
    inp.mi.mouseData = 0
    inp.mi.dwFlags = MOUSEEVENTF_MOVE
    inp.mi.time = 0
    inp.mi.dwExtraInfo = <ULONG_PTR>0  #reinterpret_cast(ulong, nullptr)
    
    SendInput(1, &inp, sizeof(INPUT))