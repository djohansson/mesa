/*
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_acceleration_structure.h"

void
mtl_acceleration_structure_get_sizes(
   mtl_device *device, const struct mtl_acceleration_structure_build *build,
   struct mtl_acceleration_structure_sizes *sizes)
{
   *sizes = (struct mtl_acceleration_structure_sizes){0};
}

void
mtl_heap_acceleration_structure_size_and_align(mtl_device *device,
                                               uint64_t *size_B,
                                               uint64_t *align_B)
{
   *align_B = 1u;
}

mtl_acceleration_structure *
mtl_new_acceleration_structure_with_size(mtl_heap *heap, uint64_t size_B,
                                         uint64_t offset_B)
{
   return NULL;
}

mtl_acceleration_structure *
mtl_new_empty_acceleration_structure(mtl_device *device)
{
   return NULL;
}

uint64_t
mtl_acceleration_structure_get_gpu_resource_id(mtl_acceleration_structure *as)
{
   return 0u;
}

void
mtl_build_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *as,
   const struct mtl_acceleration_structure_build *build,
   uint64_t scratch_address, uint64_t scratch_length)
{
}

void
mtl_refit_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *source,
   const struct mtl_acceleration_structure_build *build,
   mtl_acceleration_structure *destination, uint64_t scratch_address,
   uint64_t scratch_length)
{
}

void
mtl_copy_acceleration_structure(mtl_compute_encoder *encoder,
                                mtl_acceleration_structure *source,
                                mtl_acceleration_structure *destination,
                                bool compact)
{
}

void
mtl_write_compacted_acceleration_structure_size(mtl_compute_encoder *encoder,
                                                mtl_acceleration_structure *as,
                                                uint64_t address)
{
}
