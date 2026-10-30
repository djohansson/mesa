/*
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_acceleration_structure.h"

#include <Metal/MTL4AccelerationStructure.h>
#include <Metal/MTL4ComputeCommandEncoder.h>
#include <Metal/MTLAccelerationStructure.h>
#include <Metal/MTLAccelerationStructureCommandEncoder.h>
#include <Metal/MTLCommandQueue.h>
#include <Metal/MTLHeap.h>

static MTL4AccelerationStructureDescriptor *
mtl_new_acceleration_structure_descriptor(
   const struct mtl_acceleration_structure_build *build)
{
   @autoreleasepool {
      if (build->instances) {
         MTL4InstanceAccelerationStructureDescriptor *descriptor =
            [MTL4InstanceAccelerationStructureDescriptor new];
         descriptor.usage = (MTLAccelerationStructureUsage)build->usage;
         descriptor.instanceDescriptorType =
            MTLAccelerationStructureInstanceDescriptorTypeIndirect;
         descriptor.instanceDescriptorBuffer =
            MTL4BufferRangeMake(build->instance_address,
                                (uint64_t)build->instance_count *
                                   MTL_ACCELERATION_STRUCTURE_INSTANCE_SIZE);
         descriptor.instanceDescriptorStride =
            MTL_ACCELERATION_STRUCTURE_INSTANCE_SIZE;
         descriptor.instanceCount = build->instance_count;
         descriptor.instanceTransformationMatrixLayout =
            MTLMatrixLayoutRowMajor;
         return descriptor;
      }

      NSMutableArray<MTL4AccelerationStructureGeometryDescriptor *> *geometries =
         [NSMutableArray arrayWithCapacity:build->geometry_count];
      for (uint32_t i = 0u; i < build->geometry_count; ++i) {
         const struct mtl_acceleration_structure_geometry *g =
            &build->geometries[i];
         MTL4AccelerationStructureGeometryDescriptor *geometry;
         if (g->bounding_boxes) {
            MTL4AccelerationStructureBoundingBoxGeometryDescriptor *boxes =
               [[MTL4AccelerationStructureBoundingBoxGeometryDescriptor new]
                  autorelease];
            boxes.boundingBoxBuffer =
               MTL4BufferRangeMake(g->data_address, g->data_length);
            boxes.boundingBoxStride = g->data_stride;
            boxes.boundingBoxCount = g->primitive_count;
            geometry = boxes;
         } else {
            MTL4AccelerationStructureTriangleGeometryDescriptor *triangles =
               [[MTL4AccelerationStructureTriangleGeometryDescriptor new]
                  autorelease];
            triangles.vertexBuffer =
               MTL4BufferRangeMake(g->data_address, g->data_length);
            triangles.vertexFormat = (MTLAttributeFormat)g->vertex_format;
            triangles.vertexStride = g->data_stride;
            if (g->index_address) {
               triangles.indexBuffer =
                  MTL4BufferRangeMake(g->index_address, g->index_length);
               triangles.indexType = (MTLIndexType)g->index_type;
            }
            if (g->transform_address) {
               triangles.transformationMatrixBuffer =
                  MTL4BufferRangeMake(g->transform_address, 48u);
               triangles.transformationMatrixLayout = MTLMatrixLayoutRowMajor;
            }
            triangles.triangleCount = g->primitive_count;
            geometry = triangles;
         }
         geometry.opaque = g->opaque;
         geometry.allowDuplicateIntersectionFunctionInvocation =
            g->allow_duplicate_intersection_function_invocation;
         [geometries addObject:geometry];
      }

      MTL4PrimitiveAccelerationStructureDescriptor *descriptor =
         [MTL4PrimitiveAccelerationStructureDescriptor new];
      descriptor.usage = (MTLAccelerationStructureUsage)build->usage;
      descriptor.geometryDescriptors = geometries;
      return descriptor;
   }
}

void
mtl_acceleration_structure_get_sizes(
   mtl_device *device, const struct mtl_acceleration_structure_build *build,
   struct mtl_acceleration_structure_sizes *sizes)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTL4AccelerationStructureDescriptor *descriptor =
         [mtl_new_acceleration_structure_descriptor(build) autorelease];
      MTLAccelerationStructureSizes s =
         [dev accelerationStructureSizesWithDescriptor:descriptor];
      sizes->structure_B = s.accelerationStructureSize;
      sizes->build_scratch_B = s.buildScratchBufferSize;
      sizes->refit_scratch_B = s.refitScratchBufferSize;
   }
}

void
mtl_heap_acceleration_structure_size_and_align(mtl_device *device,
                                               uint64_t *size_B,
                                               uint64_t *align_B)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLSizeAndAlign size_and_align =
         [dev heapAccelerationStructureSizeAndAlignWithSize:*size_B];
      *size_B = size_and_align.size;
      *align_B = size_and_align.align;
   }
}

mtl_acceleration_structure *
mtl_new_acceleration_structure_with_size(mtl_heap *heap, uint64_t size_B,
                                         uint64_t offset_B)
{
   @autoreleasepool {
      id<MTLHeap> hp = (id<MTLHeap>)heap;
      return [hp newAccelerationStructureWithSize:size_B offset:offset_B];
   }
}

mtl_acceleration_structure *
mtl_new_empty_acceleration_structure(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;

      /* Metal wants a vertex buffer, even for no triangles */
      id<MTLBuffer> vertices =
         [[dev newBufferWithLength:36u
                           options:MTLResourceStorageModeShared] autorelease];
      MTLAccelerationStructureTriangleGeometryDescriptor *geometry =
         [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
      geometry.vertexBuffer = vertices;
      geometry.vertexStride = 12u;
      geometry.triangleCount = 0u;
      MTLPrimitiveAccelerationStructureDescriptor *descriptor =
         [MTLPrimitiveAccelerationStructureDescriptor descriptor];
      descriptor.geometryDescriptors = @[geometry];

      MTLAccelerationStructureSizes sizes =
         [dev accelerationStructureSizesWithDescriptor:descriptor];
      id<MTLAccelerationStructure> as =
         [dev newAccelerationStructureWithSize:sizes.accelerationStructureSize];
      id<MTLBuffer> scratch =
         [[dev newBufferWithLength:MAX(sizes.buildScratchBufferSize, 256u)
                           options:MTLResourceStorageModePrivate] autorelease];

      /* Built once, on a queue of its own */
      id<MTLCommandQueue> queue = [[dev newCommandQueue] autorelease];
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLAccelerationStructureCommandEncoder> encoder =
         [cmd accelerationStructureCommandEncoder];
      [encoder buildAccelerationStructure:as
                               descriptor:descriptor
                            scratchBuffer:scratch
                      scratchBufferOffset:0u];
      [encoder endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted) {
         [as release];
         return NULL;
      }
      return as;
   }
}

uint64_t
mtl_acceleration_structure_get_gpu_resource_id(mtl_acceleration_structure *as)
{
   @autoreleasepool {
      id<MTLAccelerationStructure> structure = (id<MTLAccelerationStructure>)as;
      return structure.gpuResourceID._impl;
   }
}

void
mtl_build_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *as,
   const struct mtl_acceleration_structure_build *build,
   uint64_t scratch_address, uint64_t scratch_length)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc =
         (id<MTL4ComputeCommandEncoder>)encoder;
      MTL4AccelerationStructureDescriptor *descriptor =
         [mtl_new_acceleration_structure_descriptor(build) autorelease];
      [enc buildAccelerationStructure:(id<MTLAccelerationStructure>)as
                           descriptor:descriptor
                        scratchBuffer:MTL4BufferRangeMake(scratch_address,
                                                          scratch_length)];
   }
}

void
mtl_refit_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *source,
   const struct mtl_acceleration_structure_build *build,
   mtl_acceleration_structure *destination, uint64_t scratch_address,
   uint64_t scratch_length)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc =
         (id<MTL4ComputeCommandEncoder>)encoder;
      MTL4AccelerationStructureDescriptor *descriptor =
         [mtl_new_acceleration_structure_descriptor(build) autorelease];
      [enc refitAccelerationStructure:(id<MTLAccelerationStructure>)source
                           descriptor:descriptor
                          destination:(id<MTLAccelerationStructure>)destination
                        scratchBuffer:MTL4BufferRangeMake(scratch_address,
                                                          scratch_length)];
   }
}

void
mtl_copy_acceleration_structure(mtl_compute_encoder *encoder,
                                mtl_acceleration_structure *source,
                                mtl_acceleration_structure *destination,
                                bool compact)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc =
         (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLAccelerationStructure> src = (id<MTLAccelerationStructure>)source;
      id<MTLAccelerationStructure> dst =
         (id<MTLAccelerationStructure>)destination;
      if (compact)
         [enc copyAndCompactAccelerationStructure:src
                          toAccelerationStructure:dst];
      else
         [enc copyAccelerationStructure:src toAccelerationStructure:dst];
   }
}

void
mtl_write_compacted_acceleration_structure_size(mtl_compute_encoder *encoder,
                                                mtl_acceleration_structure *as,
                                                uint64_t address)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc =
         (id<MTL4ComputeCommandEncoder>)encoder;
      [enc
         writeCompactedAccelerationStructureSize:(id<MTLAccelerationStructure>)as
                                        toBuffer:MTL4BufferRangeMake(address,
                                                                     8u)];
   }
}
