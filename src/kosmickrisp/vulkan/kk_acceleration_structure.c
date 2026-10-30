/*
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_acceleration_structure.h"

#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_query_pool.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "util/stack_array.h"

#include "vk_log.h"

/* Vulkan's scratch offsets are multiples of this, and so is where a top level
 * build's instances are converted to in its scratch */
#define KK_ACCELERATION_STRUCTURE_SCRATCH_ALIGNMENT 256u

uint64_t
kk_acceleration_structure_padding(struct kk_device *dev)
{
   uint64_t size_B = 1u;
   uint64_t align_B = 1u;
   mtl_heap_acceleration_structure_size_and_align(dev->mtl_handle, &size_B,
                                                  &align_B);
   return align_B > 256u ? align_B - 256u : 0u;
}

uint64_t
kk_acceleration_structure_vk_size(struct kk_device *dev, uint64_t metal_size_B)
{
   uint64_t size_B = metal_size_B;
   uint64_t align_B = 1u;
   mtl_heap_acceleration_structure_size_and_align(dev->mtl_handle, &size_B,
                                                  &align_B);
   return align64(size_B, 256u) + kk_acceleration_structure_padding(dev);
}

static enum mtl_acceleration_structure_usage
kk_acceleration_structure_usage(VkBuildAccelerationStructureFlagsKHR flags)
{
   enum mtl_acceleration_structure_usage usage =
      MTL_ACCELERATION_STRUCTURE_USAGE_NONE;
   if (flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
      usage |= MTL_ACCELERATION_STRUCTURE_USAGE_REFIT;
   if (flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
      usage |= MTL_ACCELERATION_STRUCTURE_USAGE_PREFER_FAST_BUILD;
   if (flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
      usage |= MTL_ACCELERATION_STRUCTURE_USAGE_PREFER_FAST_INTERSECTION;
   if (flags & VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR)
      usage |= MTL_ACCELERATION_STRUCTURE_USAGE_MINIMIZE_MEMORY;
   return usage;
}

static enum mtl_attribute_format
kk_acceleration_structure_vertex_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R32G32B32_SFLOAT:
      return MTL_ATTRIBUTE_FORMAT_FLOAT3;
   case VK_FORMAT_R32G32_SFLOAT:
      return MTL_ATTRIBUTE_FORMAT_FLOAT2;
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      return MTL_ATTRIBUTE_FORMAT_HALF4;
   case VK_FORMAT_R16G16_SFLOAT:
      return MTL_ATTRIBUTE_FORMAT_HALF2;
   case VK_FORMAT_R16G16B16A16_SNORM:
      return MTL_ATTRIBUTE_FORMAT_SHORT4_NORMALIZED;
   case VK_FORMAT_R16G16_SNORM:
      return MTL_ATTRIBUTE_FORMAT_SHORT2_NORMALIZED;
   default:
      UNREACHABLE("Unsupported acceleration structure vertex format");
   }
}

static const VkAccelerationStructureGeometryKHR *
kk_build_geometry(const VkAccelerationStructureBuildGeometryInfoKHR *info,
                  uint32_t index)
{
   return info->pGeometries ? &info->pGeometries[index]
                            : info->ppGeometries[index];
}

/* Where a top level build converts its instances to, at the start of its
 * scratch (see libkk_convert_acceleration_structure_instances) */
static uint64_t
kk_instances_scratch_size(uint32_t instance_count)
{
   return align64(
      (uint64_t)instance_count * MTL_ACCELERATION_STRUCTURE_INSTANCE_SIZE,
      KK_ACCELERATION_STRUCTURE_SCRATCH_ALIGNMENT);
}

/* A bottom level build's geometries in Metal's terms, at the addresses its
 * ranges say. Without ranges (sizes are queried before there is anything to
 * build), the counts are the maximum counts and the addresses stand-ins, so
 * that a geometry is described as it will be built (indexed, transformed). */
static void
kk_fill_geometries(const VkAccelerationStructureBuildGeometryInfoKHR *info,
                   const VkAccelerationStructureBuildRangeInfoKHR *ranges,
                   const uint32_t *max_primitive_counts,
                   uint64_t fallback_address,
                   struct mtl_acceleration_structure_geometry *geometries)
{
   for (uint32_t i = 0u; i < info->geometryCount; ++i) {
      const VkAccelerationStructureGeometryKHR *g = kk_build_geometry(info, i);
      const VkAccelerationStructureBuildRangeInfoKHR range =
         ranges ? ranges[i]
                : (VkAccelerationStructureBuildRangeInfoKHR){
                     .primitiveCount = max_primitive_counts[i]};
      struct mtl_acceleration_structure_geometry *m = &geometries[i];

      *m = (struct mtl_acceleration_structure_geometry){
         .opaque = (g->flags & VK_GEOMETRY_OPAQUE_BIT_KHR) != 0,
         .allow_duplicate_intersection_function_invocation =
            !(g->flags & VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR),
         .primitive_count = range.primitiveCount,
      };

      if (g->geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
         const VkAccelerationStructureGeometryTrianglesDataKHR *t =
            &g->geometry.triangles;
         uint64_t vertices =
            ranges ? t->vertexData.deviceAddress +
                        (uint64_t)range.firstVertex * t->vertexStride
                   : fallback_address;
         m->vertex_format =
            kk_acceleration_structure_vertex_format(t->vertexFormat);
         m->data_stride = t->vertexStride;
         if (t->indexType != VK_INDEX_TYPE_NONE_KHR) {
            const uint32_t index_size =
               t->indexType == VK_INDEX_TYPE_UINT16 ? 2u : 4u;
            m->index_address =
               ranges ? t->indexData.deviceAddress + range.primitiveOffset
                      : fallback_address;
            m->index_type = t->indexType == VK_INDEX_TYPE_UINT16
                               ? MTL_INDEX_TYPE_UINT16
                               : MTL_INDEX_TYPE_UINT32;
            m->index_length = (uint64_t)range.primitiveCount * 3u * index_size;
            m->data_length = ((uint64_t)t->maxVertex + 1u) * t->vertexStride;
         } else {
            if (ranges)
               vertices += range.primitiveOffset;
            m->data_length =
               (uint64_t)range.primitiveCount * 3u * t->vertexStride;
         }
         m->data_address = vertices;
         /* Sizes are queried with the transform's address null or not */
         if (t->transformData.deviceAddress) {
            m->transform_address =
               ranges ? t->transformData.deviceAddress + range.transformOffset
                      : fallback_address;
         }
      } else {
         assert(g->geometryType == VK_GEOMETRY_TYPE_AABBS_KHR);
         const VkAccelerationStructureGeometryAabbsDataKHR *a =
            &g->geometry.aabbs;
         m->bounding_boxes = true;
         m->data_address = ranges
                              ? a->data.deviceAddress + range.primitiveOffset
                              : fallback_address;
         m->data_stride = a->stride;
         m->data_length = (uint64_t)range.primitiveCount * a->stride;
      }

      /* Metal wants an address even for no primitives */
      if (m->data_address == 0u)
         m->data_address = fallback_address;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateAccelerationStructureKHR(
   VkDevice device, const VkAccelerationStructureCreateInfoKHR *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkAccelerationStructureKHR *pAccelerationStructure)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_buffer, buffer, pCreateInfo->buffer);

   /* Placed in the buffer's heap: memory imported from the host has none */
   if (buffer->metal.heap == NULL)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "acceleration structure in memory without a heap");

   uint64_t size_B = 1u;
   uint64_t align_B = 1u;
   mtl_heap_acceleration_structure_size_and_align(dev->mtl_handle, &size_B,
                                                  &align_B);
   const uint64_t start = buffer->metal.heap_offset + pCreateInfo->offset;
   const uint64_t placed = align64(start, align_B);
   if (placed - start >= pCreateInfo->size)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "acceleration structure too small to place");

   struct kk_acceleration_structure *as =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*as),
                       VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR);
   if (as == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   as->mtl_handle = mtl_new_acceleration_structure_with_size(
      buffer->metal.heap, pCreateInfo->size - (placed - start), placed);
   if (as->mtl_handle == NULL) {
      vk_object_free(&dev->vk, pAllocator, as);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   as->gpu_resource_id =
      mtl_acceleration_structure_get_gpu_resource_id(as->mtl_handle);

   *pAccelerationStructure = kk_acceleration_structure_to_handle(as);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyAccelerationStructureKHR(
   VkDevice device, VkAccelerationStructureKHR accelerationStructure,
   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_acceleration_structure, as, accelerationStructure);

   if (as == NULL)
      return;

   mtl_release(as->mtl_handle);
   vk_object_free(&dev->vk, pAllocator, as);
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL
kk_GetAccelerationStructureDeviceAddressKHR(
   VkDevice device, const VkAccelerationStructureDeviceAddressInfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_acceleration_structure, as, pInfo->accelerationStructure);
   return as->gpu_resource_id;
}

VKAPI_ATTR void VKAPI_CALL
kk_GetAccelerationStructureBuildSizesKHR(
   VkDevice device, VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   VK_FROM_HANDLE(kk_device, dev, device);

   /* A stand-in address for what is described but not built yet */
   const uint64_t stand_in = 0x10000u;

   struct mtl_acceleration_structure_build build = {
      .usage = kk_acceleration_structure_usage(pBuildInfo->flags),
   };
   uint64_t instances_scratch = 0u;

   STACK_ARRAY(struct mtl_acceleration_structure_geometry, geometries,
               pBuildInfo->geometryCount);
   if (pBuildInfo->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
      build.instances = true;
      build.instance_address = stand_in;
      build.instance_count = pMaxPrimitiveCounts[0];
      instances_scratch = kk_instances_scratch_size(build.instance_count);
   } else {
      kk_fill_geometries(pBuildInfo, NULL, pMaxPrimitiveCounts, stand_in,
                         geometries);
      build.geometry_count = pBuildInfo->geometryCount;
      build.geometries = geometries;
   }

   struct mtl_acceleration_structure_sizes sizes;
   mtl_acceleration_structure_get_sizes(dev->mtl_handle, &build, &sizes);
   STACK_ARRAY_FINISH(geometries);

   const uint64_t scratch_alignment =
      KK_ACCELERATION_STRUCTURE_SCRATCH_ALIGNMENT;
   pSizeInfo->accelerationStructureSize =
      kk_acceleration_structure_vk_size(dev, sizes.structure_B);
   pSizeInfo->buildScratchSize =
      instances_scratch +
      align64(MAX2(sizes.build_scratch_B, 1u), scratch_alignment);
   pSizeInfo->updateScratchSize =
      instances_scratch +
      align64(MAX2(sizes.refit_scratch_B, 1u), scratch_alignment);
}

VKAPI_ATTR void VKAPI_CALL
kk_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice device, const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   /* Nothing is serialized */
   *pCompatibility = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   for (uint32_t i = 0u; i < infoCount; ++i) {
      const VkAccelerationStructureBuildGeometryInfoKHR *info = &pInfos[i];
      VK_FROM_HANDLE(kk_acceleration_structure, dst,
                     info->dstAccelerationStructure);
      VK_FROM_HANDLE(kk_acceleration_structure, src,
                     info->srcAccelerationStructure);
      const bool update =
         info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
      uint64_t scratch = info->scratchData.deviceAddress;

      struct mtl_acceleration_structure_build build = {
         .usage = kk_acceleration_structure_usage(info->flags),
      };

      STACK_ARRAY(struct mtl_acceleration_structure_geometry, geometries,
                  info->geometryCount);
      if (info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
         const VkAccelerationStructureGeometryInstancesDataKHR *instances =
            &kk_build_geometry(info, 0u)->geometry.instances;
         const VkAccelerationStructureBuildRangeInfoKHR *range =
            &ppBuildRangeInfos[i][0];

         build.instances = true;
         build.instance_address = scratch;
         build.instance_count = range->primitiveCount;
         scratch += kk_instances_scratch_size(range->primitiveCount);

         if (range->primitiveCount > 0u) {
            libkk_convert_acceleration_structure_instances(
               cmd, kk_grid_1d(range->primitiveCount), false,
               build.instance_address,
               instances->data.deviceAddress + range->primitiveOffset,
               dev->empty_acceleration_structure_id, instances->arrayOfPointers,
               range->primitiveCount);
            mtl_barrier_after_encoder_stages(cs_get_compute(cmd),
                                             MTL_STAGE_DISPATCH,
                                             MTL_STAGE_ACCELERATION_STRUCTURE);
         }
      } else {
         kk_fill_geometries(info, ppBuildRangeInfos[i], NULL, scratch,
                            geometries);
         build.geometry_count = info->geometryCount;
         build.geometries = geometries;
      }

      mtl_compute_encoder *encoder = cs_get_compute(cmd);
      if (update) {
         mtl_refit_acceleration_structure(encoder, src->mtl_handle, &build,
                                          src == dst ? NULL : dst->mtl_handle,
                                          scratch, UINT64_MAX);
      } else {
         mtl_build_acceleration_structure(encoder, dst->mtl_handle, &build,
                                          scratch, UINT64_MAX);
      }
      STACK_ARRAY_FINISH(geometries);
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBuildAccelerationStructuresIndirectKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkDeviceAddress *pIndirectDeviceAddresses,
   const uint32_t *pIndirectStrides,
   const uint32_t *const *ppMaxPrimitiveCounts)
{
   UNREACHABLE("accelerationStructureIndirectBuild is not supported");
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(kk_acceleration_structure, src, pInfo->src);
   VK_FROM_HANDLE(kk_acceleration_structure, dst, pInfo->dst);

   switch (pInfo->mode) {
   case VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR:
   case VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR:
      mtl_copy_acceleration_structure(
         cs_get_compute(cmd), src->mtl_handle, dst->mtl_handle,
         pInfo->mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR);
      break;
   default:
      UNREACHABLE("Acceleration structures aren't serialized");
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdCopyAccelerationStructureToMemoryKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   UNREACHABLE("Acceleration structures aren't serialized");
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdCopyMemoryToAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   UNREACHABLE("Acceleration structures aren't serialized");
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdWriteAccelerationStructuresPropertiesKHR(
   VkCommandBuffer commandBuffer, uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType, VkQueryPool queryPool, uint32_t firstQuery)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(kk_query_pool, pool, queryPool);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   if (queryType != VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
      UNREACHABLE("Unsupported acceleration structure property");

   const uint64_t padding = kk_acceleration_structure_padding(dev);
   for (uint32_t i = 0u; i < accelerationStructureCount; ++i) {
      VK_FROM_HANDLE(kk_acceleration_structure, as, pAccelerationStructures[i]);

      /* Metal writes the size, which the report then takes as Vulkan sizes
       * a structure */
      struct kk_ptr size = kk_pool_alloc(cmd, sizeof(uint64_t), 8u);
      if (unlikely(!size.gpu))
         return;

      mtl_compute_encoder *encoder = cs_get_compute(cmd);
      mtl_write_compacted_acceleration_structure_size(encoder, as->mtl_handle,
                                                      size.gpu);
      mtl_barrier_after_encoder_stages(
         encoder, MTL_STAGE_ACCELERATION_STRUCTURE, MTL_STAGE_DISPATCH);
      libkk_write_acceleration_structure_size(
         cmd, kk_grid_1d(1u), false,
         kk_query_report_address(dev, pool, firstQuery + i), size.gpu, padding);
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_BuildAccelerationStructuresKHR(
   VkDevice device, VkDeferredOperationKHR deferredOperation,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   UNREACHABLE("accelerationStructureHostCommands is not supported");
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CopyAccelerationStructureKHR(VkDevice device,
                                VkDeferredOperationKHR deferredOperation,
                                const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   UNREACHABLE("accelerationStructureHostCommands is not supported");
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CopyAccelerationStructureToMemoryKHR(
   VkDevice device, VkDeferredOperationKHR deferredOperation,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   UNREACHABLE("accelerationStructureHostCommands is not supported");
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CopyMemoryToAccelerationStructureKHR(
   VkDevice device, VkDeferredOperationKHR deferredOperation,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   UNREACHABLE("accelerationStructureHostCommands is not supported");
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_WriteAccelerationStructuresPropertiesKHR(
   VkDevice device, uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType, size_t dataSize, void *pData, size_t stride)
{
   UNREACHABLE("accelerationStructureHostCommands is not supported");
   return VK_ERROR_FEATURE_NOT_PRESENT;
}
