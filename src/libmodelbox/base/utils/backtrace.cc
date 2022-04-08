/* Android does not provide execinfo.h. We implement backtrace and
 * backtrace_symbols in this file. */

#include <unwind.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ANDROID

int backtrace(void **buffer, int size);
char **backtrace_symbols(void *const *buffer, int size);

typedef struct _BacktraceCtrl
{
    void** buffer;
    int    size;
    int    index;
} BacktraceCtrl;

static _Unwind_Reason_Code unwind_callback(struct _Unwind_Context* context, void* arg)
{
    BacktraceCtrl *ctrl;
    unsigned long  pc;
    
    ctrl = (BacktraceCtrl *)arg;
    if (ctrl->index >= ctrl->size)
    {
        return _URC_END_OF_STACK;
    }
    
    pc = _Unwind_GetIP(context);
    if (pc) 
    {
        ctrl->buffer[ctrl->index] = (void *)pc;
        ctrl->index++;
    }
    return _URC_NO_REASON;
}

int backtrace(void **buffer, int size)
{
    BacktraceCtrl ctrl;
    
    ctrl.buffer = buffer;
    ctrl.size   = size;
    ctrl.index  = 0;
    
    _Unwind_Backtrace(unwind_callback, &ctrl);
    
    return ctrl.index;
}

#define MEM_BLK_SiZE   4096

char **backtrace_symbols(void *const *buffer, int size)
{   
    char       *symbols;
    char       *ptr;
    int         pos;
    int         len;

    int         index;
    char       *addr;
    Dl_info     info;
    
    if (size <= 0)
    {
        return NULL;
    }
    
    symbols = (char *)malloc(size * sizeof(void *) + MEM_BLK_SiZE);
    if (symbols == NULL)
    {
        return NULL;
    }
    
    memset(symbols, 0, sizeof(void *));
    
    pos = size * sizeof(void *);
    len = pos + MEM_BLK_SiZE;
    
    for (index = 0; index < size; index++) 
    {
        addr = (char *)buffer[index];

        /* Need more space? */
        if (len - pos < MEM_BLK_SiZE / 4)
        {
            ptr = (char *)realloc(symbols, len + MEM_BLK_SiZE);
            
            /* If realloc() fails, the original block is left untouched; it is not
               freed or moved. */
            if (ptr == NULL)
            {
                break;
            }
            
            symbols = ptr;
            len += MEM_BLK_SiZE;
        }
        
        *((char **)symbols + index) = symbols + pos;
        
        /* If  the address specified in addr could not be matched to a shared 
           object, then these functions return 0. */
        if (!dladdr(addr, &info) || info.dli_fname == NULL)
        {
            pos += snprintf(symbols + pos, len - pos, "Unknown(+0) [%p]", addr) + 1;
            continue;
        }
        
        /* If no symbol matching addr could be found, then dli_sname and 
           dli_saddr are set to NULL. */
        if (info.dli_sname == NULL || info.dli_saddr == NULL) 
        {
            pos += snprintf(symbols + pos, len - pos, "%s(+%lx) [%p]", info.dli_fname, 
                addr - (char *)info.dli_fbase, addr) + 1;
            continue;
        }
        
        pos += snprintf(symbols + pos, len - pos, "%s(%s+%lx) [%p]", info.dli_fname, 
            info.dli_sname, addr - (char *)info.dli_saddr, addr) + 1;
    }
    
    return (char **)symbols;
}

#endif /* ANDROID */

#ifdef __cplusplus
}
#endif

