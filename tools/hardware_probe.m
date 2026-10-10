#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreML/CoreML.h>
int main(void) { @autoreleasepool {
 NSArray<id<MTLDevice>> *devices=MTLCopyAllDevices();
 NSMutableArray *names=[NSMutableArray array];
 for(id<MTLDevice> d in devices) [names addObject:@{@"name":d.name,@"unified_memory":@(d.hasUnifiedMemory)}];
 NSMutableDictionary *report=[@{@"metal_device_count":@(devices.count),@"metal_devices":names,@"default_metal_device":MTLCreateSystemDefaultDevice().name ?: @"none"} mutableCopy];
 if (@available(macOS 14.0,*)) { NSMutableArray *compute=[NSMutableArray array]; for(id d in MLModel.availableComputeDevices) [compute addObject:NSStringFromClass([d class])]; report[@"coreml_compute_devices"]=compute; }
 NSData *data=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
 puts([[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding].UTF8String);
 } return 0; }
