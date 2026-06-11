// Build keryx-opencl.cl for a given arch + defines, dump the compiled binary,
// and report the kernel's VGPR/SGPR usage (so we can see if it is occupancy-bound).
// cc kdump.c -I/opt/rocm-6.4.0/include -L<ocllib> -lOpenCL -o kdump
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(const char *p, size_t *n){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}fseek(f,0,SEEK_END);long s=ftell(f);fseek(f,0,SEEK_SET);char*b=malloc(s+1);if(fread(b,1,s,f)!=(size_t)s){perror("fread");exit(1);}b[s]=0;*n=s;fclose(f);return b;}

int main(int argc,char**argv){
    const char*src=argc>1?argv[1]:"resources/keryx-opencl.cl";
    const char*arch=argc>2?argv[2]:"gfx1102";
    const char*defs=argc>3?argv[3]:"-D __FORCE_AMD_V_DOT8_U32_U4__=1 ";
    const char*outbin=argc>4?argv[4]:"/tmp/keryx.bin";

    cl_uint np=0;clGetPlatformIDs(0,NULL,&np);cl_platform_id*pl=calloc(np,sizeof(*pl));clGetPlatformIDs(np,pl,NULL);
    cl_device_id dev=0;char buf[256];
    for(cl_uint p=0;p<np&&!dev;p++){cl_uint nd=0;if(clGetDeviceIDs(pl[p],CL_DEVICE_TYPE_ALL,0,NULL,&nd)!=CL_SUCCESS)continue;cl_device_id*d=calloc(nd,sizeof(*d));clGetDeviceIDs(pl[p],CL_DEVICE_TYPE_ALL,nd,d,NULL);for(cl_uint i=0;i<nd;i++){clGetDeviceInfo(d[i],CL_DEVICE_NAME,sizeof(buf),buf,NULL);if(strstr(buf,arch)){dev=d[i];break;}}free(d);}
    if(!dev){fprintf(stderr,"no %s\n",arch);return 1;}
    cl_int err;cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    size_t sl;char*s=slurp(src,&sl);const char*ss[1]={s};
    cl_program pr=clCreateProgramWithSource(ctx,1,ss,&sl,&err);
    char opts[600];snprintf(opts,sizeof(opts),"%s-cl-mad-enable -cl-finite-math-only -cl-std=CL2.0 -DAMD_ACCELERATED_PARALLEL_PROCESSING -D OPENCL_PLATFORM_AMD -D __%s__ ",defs,arch);
    printf("arch=%s opts=%s\n",arch,opts);
    err=clBuildProgram(pr,1,&dev,opts,NULL,NULL);
    size_t lg=0;clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&lg);
    if(lg>1){char*l=malloc(lg);clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,lg,l,NULL);printf("log:%s\n",l);}
    if(err){fprintf(stderr,"build err %d\n",err);return 1;}
    // dump binary
    size_t bsz=0;clGetProgramInfo(pr,CL_PROGRAM_BINARY_SIZES,sizeof(bsz),&bsz,NULL);
    unsigned char*bin=malloc(bsz);unsigned char*bins[1]={bin};
    clGetProgramInfo(pr,CL_PROGRAM_BINARIES,sizeof(bins),bins,NULL);
    FILE*o=fopen(outbin,"wb");fwrite(bin,1,bsz,o);fclose(o);
    printf("wrote %zu bytes -> %s\n",bsz,outbin);
    // kernel work-group info
    cl_kernel k=clCreateKernel(pr,"heavy_hash",&err);
    if(!err){
        size_t wgs=0;clGetKernelWorkGroupInfo(k,dev,CL_KERNEL_WORK_GROUP_SIZE,sizeof(wgs),&wgs,NULL);
        cl_ulong pmem=0;clGetKernelWorkGroupInfo(k,dev,CL_KERNEL_PRIVATE_MEM_SIZE,sizeof(pmem),&pmem,NULL);
        cl_ulong lmem=0;clGetKernelWorkGroupInfo(k,dev,CL_KERNEL_LOCAL_MEM_SIZE,sizeof(lmem),&lmem,NULL);
        printf("heavy_hash: max_wg=%zu private_mem=%llu local_mem=%llu\n",wgs,(unsigned long long)pmem,(unsigned long long)lmem);
    }
    return 0;
}
