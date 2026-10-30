/*
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_ACCELERATION_STRUCTURE_H
#define MTL_ACCELERATION_STRUCTURE_H 1

#include "mtl_types.h"

#include <stdbool.h>
#include <stdint.h>

/* A bottom level structure's geometry: triangles or bounding boxes */
struct mtl_acceleration_structure_geometry {
   bool bounding_boxes;
   bool opaque;
   bool allow_duplicate_intersection_function_invocation;
   uint32_t primitive_count;

   /* The vertices (triangles) or the boxes (MTLAxisAlignedBoundingBox) */
   uint64_t data_address;
   uint64_t data_length;
   uint32_t data_stride;
   enum mtl_attribute_format vertex_format;

   /* Optional (0): the indices, and a row major 3x4 transform */
   uint64_t index_address;
   uint64_t index_length;
   enum mtl_index_type index_type;
   uint64_t transform_address;
};

/* What a structure is built of: the geometries of a bottom level structure,
 * or the instances of a top level one (MTLIndirectAccelerationStructure
 * InstanceDescriptor, their transforms row major) */
struct mtl_acceleration_structure_build {
   enum mtl_acceleration_structure_usage usage;

   uint32_t geometry_count;
   const struct mtl_acceleration_structure_geometry *geometries;

   bool instances;
   uint64_t instance_address;
   uint32_t instance_count;
};

struct mtl_acceleration_structure_sizes {
   uint64_t structure_B;
   uint64_t build_scratch_B;
   uint64_t refit_scratch_B;
};

/* Size of an MTLIndirectAccelerationStructureInstanceDescriptor */
#define MTL_ACCELERATION_STRUCTURE_INSTANCE_SIZE 72u

void mtl_acceleration_structure_get_sizes(
   mtl_device *device, const struct mtl_acceleration_structure_build *build,
   struct mtl_acceleration_structure_sizes *sizes);

void mtl_heap_acceleration_structure_size_and_align(mtl_device *device,
                                                    uint64_t *size_B,
                                                    uint64_t *align_B);

/* Placed in a heap at an offset aligned as
 * mtl_heap_acceleration_structure_size_and_align says */
mtl_acceleration_structure *
mtl_new_acceleration_structure_with_size(mtl_heap *heap, uint64_t size_B,
                                         uint64_t offset_B);

/* A bottom level structure of nothing, built before it returns, for what
 * references none (Vulkan's inactive instances). Not in a heap: it needs to be
 * made resident on its own */
mtl_acceleration_structure *
mtl_new_empty_acceleration_structure(mtl_device *device);

uint64_t
mtl_acceleration_structure_get_gpu_resource_id(mtl_acceleration_structure *as);

void mtl_build_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *as,
   const struct mtl_acceleration_structure_build *build,
   uint64_t scratch_address, uint64_t scratch_length);

/* In place where destination is NULL */
void mtl_refit_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *source,
   const struct mtl_acceleration_structure_build *build,
   mtl_acceleration_structure *destination, uint64_t scratch_address,
   uint64_t scratch_length);

void mtl_copy_acceleration_structure(mtl_compute_encoder *encoder,
                                     mtl_acceleration_structure *source,
                                     mtl_acceleration_structure *destination,
                                     bool compact);

/* Writes the size as a 64-bit value */
void
mtl_write_compacted_acceleration_structure_size(mtl_compute_encoder *encoder,
                                                mtl_acceleration_structure *as,
                                                uint64_t address);

#endif /* MTL_ACCELERATION_STRUCTURE_H */
