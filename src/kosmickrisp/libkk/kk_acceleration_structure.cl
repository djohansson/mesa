/*
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl_vk.h"

/*
 * Converts a top level build's instances, Vulkan's VkAccelerationStructure
 * InstanceKHR (an array of them, or of pointers to them), into Metal's
 * MTLIndirectAccelerationStructureInstanceDescriptor: the transform as it is
 * (both row major 3x4), the flags as the options (the same bits), the mask,
 * the shader binding table offset as the intersection function table's, the
 * custom index as the user id, and the referenced structure's Metal resource
 * id (its Vulkan address). An instance referencing none is inactive: it
 * references an empty structure, with a mask nothing matches.
 */
KERNEL(32)
libkk_convert_acceleration_structure_instances(global uint32_t *dst,
                                               global uint32_t *src,
                                               uint64_t empty_structure,
                                               uint32_t array_of_pointers,
                                               uint32_t count)
{
   uint i = cl_global_id.x;
   if (i >= count)
      return;

   global uint32_t *instance =
      array_of_pointers ? (global uint32_t *)((global uint64_t *)src)[i]
                        : src + i * 16u;
   global uint32_t *descriptor = dst + i * 18u;

   for (uint j = 0u; j < 12u; ++j)
      descriptor[j] = instance[j];

   uint32_t custom_index_and_mask = instance[12];
   uint32_t sbt_offset_and_flags = instance[13];
   uint64_t structure = instance[14] | ((uint64_t)instance[15] << 32);

   descriptor[12] = sbt_offset_and_flags >> 24;
   descriptor[13] = structure != 0u ? custom_index_and_mask >> 24 : 0u;
   descriptor[14] = sbt_offset_and_flags & 0xffffffu;
   descriptor[15] = custom_index_and_mask & 0xffffffu;

   if (structure == 0u)
      structure = empty_structure;
   descriptor[16] = (uint32_t)structure;
   descriptor[17] = (uint32_t)(structure >> 32);
}

/*
 * Writes a compacted size query's report: the size Metal wrote, as Vulkan
 * sizes a structure (see kk_acceleration_structure_vk_size), which makes the
 * query available.
 */
KERNEL(1)
libkk_write_acceleration_structure_size(global uint64_t *report,
                                        global uint64_t *size, uint64_t padding)
{
   *report = ((*size + 255u) & ~(uint64_t)255u) + padding;
}
