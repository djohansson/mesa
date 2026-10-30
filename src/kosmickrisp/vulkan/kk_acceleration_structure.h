/*
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_ACCELERATION_STRUCTURE_H
#define KK_ACCELERATION_STRUCTURE_H 1

#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "vk_object.h"

struct kk_device;

/* A Metal acceleration structure placed in its buffer's memory (its heap).
 * Metal places one at a multiple of its heap alignment (16 KB), Vulkan at a
 * multiple of 256 bytes in its buffer: the sizes Vulkan is given leave room to
 * move it to the next (kk_acceleration_structure_vk_size). Its device address,
 * which top level structures reference it by, and its descriptor are its
 * Metal resource id, which the shaders take it by. */
struct kk_acceleration_structure {
   struct vk_object_base base;

   mtl_acceleration_structure *mtl_handle;
   uint64_t gpu_resource_id;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_acceleration_structure, base,
                               VkAccelerationStructureKHR,
                               VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR)

/* What Metal places in a heap past its alignment from where Vulkan places a
 * structure */
uint64_t kk_acceleration_structure_padding(struct kk_device *dev);

/* The size Vulkan is given for a structure Metal sizes so */
uint64_t kk_acceleration_structure_vk_size(struct kk_device *dev,
                                           uint64_t metal_size_B);

#endif /* KK_ACCELERATION_STRUCTURE_H */
