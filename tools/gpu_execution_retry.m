#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define CL_SILENCE_DEPRECATION
#import <OpenCL/opencl.h>
#include <math.h>

static NSDictionary *metalProbe(void) {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSMutableArray *names = [NSMutableArray array];
    for (id<MTLDevice> d in devices) [names addObject:d.name];
    NSMutableDictionary *r = [@{@"enumerated_devices":names, @"device_count":@(devices.count),
                                @"gpu_execution_confirmed":@NO} mutableCopy];
    if (!device) { r[@"stage"]=@"device_creation"; r[@"error"]=@"MTLCreateSystemDefaultDevice returned nil"; return r; }
    r[@"device"]=device.name;
    NSError *error=nil;
    NSString *source=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void add_arrays(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i<256) c[i]=a[i]+b[i]; }";
    id<MTLLibrary> library=[device newLibraryWithSource:source options:nil error:&error];
    if (!library) { r[@"stage"]=@"shader_compilation"; r[@"error"]=error.localizedDescription ?: @"no library"; return r; }
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"add_arrays"] error:&error];
    if (!pipeline) { r[@"stage"]=@"pipeline_creation"; r[@"error"]=error.localizedDescription ?: @"no pipeline"; return r; }
    id<MTLBuffer> a=[device newBufferWithLength:256*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b=[device newBufferWithLength:256*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> c=[device newBufferWithLength:256*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[device newCommandQueue];
    if (!a || !b || !c || !queue) { r[@"stage"]=@"resource_creation"; r[@"error"]=@"buffer or command queue creation failed"; return r; }
    for (int i=0;i<256;i++) { ((float*)a.contents)[i]=i; ((float*)b.contents)[i]=2*i; ((float*)c.contents)[i]=-1; }
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if (!command || !encoder) { r[@"stage"]=@"command_creation"; r[@"error"]=@"command buffer or encoder creation failed"; return r; }
    [encoder setComputePipelineState:pipeline]; [encoder setBuffer:a offset:0 atIndex:0];
    [encoder setBuffer:b offset:0 atIndex:1]; [encoder setBuffer:c offset:0 atIndex:2];
    NSUInteger width=MIN((NSUInteger)16,pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    BOOL correct=command.status==MTLCommandBufferStatusCompleted;
    for (int i=0;i<256 && correct;i++) correct=((float*)c.contents)[i]==3*i;
    r[@"stage"]=@"execution"; r[@"command_status"]=@(command.status);
    r[@"gpu_execution_confirmed"]=@(command.status==MTLCommandBufferStatusCompleted);
    r[@"output_correct"]=@(correct);
    if (command.error) r[@"error"]=command.error.localizedDescription;
    return r;
}

static NSDictionary *openclProbe(void) {
    NSMutableDictionary *r=[@{@"requested_type":@"GPU",@"gpu_execution_confirmed":@NO} mutableCopy];
    cl_uint count=0; cl_int status=clGetPlatformIDs(0,NULL,&count);
    r[@"platform_status"]=@(status); r[@"platform_count"]=@(count);
    if (status!=CL_SUCCESS || count==0) {r[@"stage"]=@"platform_enumeration"; return r;}
    cl_platform_id *platforms=calloc(count,sizeof(cl_platform_id));
    status=clGetPlatformIDs(count,platforms,NULL);
    if (status!=CL_SUCCESS) {free(platforms);r[@"stage"]=@"platform_enumeration";r[@"error_code"]=@(status);return r;}
    cl_device_id device=NULL; NSMutableArray *statuses=[NSMutableArray array];
    for (cl_uint p=0;p<count;p++) {
        cl_uint n=0; cl_int s=clGetDeviceIDs(platforms[p],CL_DEVICE_TYPE_GPU,0,NULL,&n);
        NSMutableDictionary *entry=[@{@"count_query_status":@(s),@"gpu_count":@(n)} mutableCopy];
        cl_device_id candidates[16]={0};cl_uint fixedCount=0;
        cl_int fixed=clGetDeviceIDs(platforms[p],CL_DEVICE_TYPE_GPU,16,candidates,&fixedCount);
        entry[@"fixed_array_query_status"]=@(fixed);entry[@"fixed_array_gpu_count"]=@(fixedCount);
        char pname[256]={0};cl_int info=clGetPlatformInfo(platforms[p],CL_PLATFORM_NAME,sizeof(pname),pname,NULL);
        entry[@"platform_name_query_status"]=@(info);if(info==CL_SUCCESS)entry[@"platform_name"]=@(pname);
        [statuses addObject:entry];
        if (fixed==CL_SUCCESS && fixedCount>0) {device=candidates[0];break;}
    }
    free(platforms);r[@"device_enumeration"]=statuses;
    if (!device) {r[@"stage"]=@"gpu_device_enumeration";return r;}
    char name[256]={0};clGetDeviceInfo(device,CL_DEVICE_NAME,sizeof(name),name,NULL);r[@"device"]=@(name);
    cl_context context=clCreateContext(NULL,1,&device,NULL,NULL,&status);
    if (!context || status!=CL_SUCCESS) {r[@"stage"]=@"context_creation";r[@"error_code"]=@(status);return r;}
    cl_command_queue queue=clCreateCommandQueue(context,device,0,&status);
    if (!queue || status!=CL_SUCCESS) {clReleaseContext(context);r[@"stage"]=@"queue_creation";r[@"error_code"]=@(status);return r;}
    const char *source="__kernel void add_arrays(__global const float* a,__global const float* b,__global float* c){int i=get_global_id(0);c[i]=a[i]+b[i];}";
    cl_program program=clCreateProgramWithSource(context,1,&source,NULL,&status);
    if (!program || status!=CL_SUCCESS) {clReleaseCommandQueue(queue);clReleaseContext(context);r[@"stage"]=@"program_creation";r[@"error_code"]=@(status);return r;}
    status=clBuildProgram(program,1,&device,NULL,NULL,NULL);
    if (status!=CL_SUCCESS) {clReleaseProgram(program);clReleaseCommandQueue(queue);clReleaseContext(context);r[@"stage"]=@"program_build";r[@"error_code"]=@(status);return r;}
    cl_kernel kernel=clCreateKernel(program,"add_arrays",&status);
    float a[256],b[256],out[256];for(int i=0;i<256;i++){a[i]=i;b[i]=2*i;out[i]=-1;}
    cl_mem ba=NULL,bb=NULL,bc=NULL; cl_int step=CL_SUCCESS;
    if (!kernel || status!=CL_SUCCESS) goto cleanup;
    ba=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(a),a,&status); if(status!=CL_SUCCESS)goto cleanup;
    bb=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(b),b,&status); if(status!=CL_SUCCESS)goto cleanup;
    bc=clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(out),NULL,&status); if(status!=CL_SUCCESS)goto cleanup;
    status=clSetKernelArg(kernel,0,sizeof(ba),&ba); if(status!=CL_SUCCESS)goto cleanup;
    status=clSetKernelArg(kernel,1,sizeof(bb),&bb); if(status!=CL_SUCCESS)goto cleanup;
    status=clSetKernelArg(kernel,2,sizeof(bc),&bc); if(status!=CL_SUCCESS)goto cleanup;
    size_t global=256;status=clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,NULL,0,NULL,NULL); if(status!=CL_SUCCESS)goto cleanup;
    status=clFinish(queue);if(status!=CL_SUCCESS)goto cleanup;
    r[@"gpu_execution_confirmed"]=@YES;
    status=clEnqueueReadBuffer(queue,bc,CL_TRUE,0,sizeof(out),out,0,NULL,NULL);
    if(status==CL_SUCCESS) {BOOL correct=YES;for(int i=0;i<256;i++)if(out[i]!=3*i)correct=NO;r[@"output_correct"]=@(correct);}
cleanup:
    step=status;r[@"stage"]=@"gpu_execution_attempt";r[@"status"]=@(step);
    if(ba)clReleaseMemObject(ba);if(bb)clReleaseMemObject(bb);if(bc)clReleaseMemObject(bc);
    if(kernel)clReleaseKernel(kernel);clReleaseProgram(program);clReleaseCommandQueue(queue);clReleaseContext(context);
    return r;
}

int main(void) { @autoreleasepool {
    NSDictionary *report=@{@"metal":metalProbe(),@"opencl":openclProbe(),
                           @"scope":@"Explicit GPU vector-add attempt; no CPU device substitution"};
    NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
    puts([[NSString alloc]initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
}return 0;}
