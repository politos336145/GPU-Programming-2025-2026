// ============================================================================
// vulkan_renderer_billboard.inl
//
// Billboard rendering: Angry Santa (top of slope) + Christmas Village (bottom).
// The village switches to the "Destroyed" version when the avalanche arrives.
//
// HOW TO USE:
//   #include "vulkan_renderer_billboard.inl"
//   at the END of vulkan_renderer.cu, just before #endif // ENABLE_VULKAN
//
// This file is included into the same translation unit as vulkan_renderer.cu,
// giving it direct access to:
//   - g_ctx              (renderer state)
//   - findMemoryType()   (memory helper)
//   - readSPIRV()        (shader loader)
//   - createShaderModule()
//   - stbi_load / stbi_image_free (stb_image.h already included)
//
// Forward declarations to add in the "Forward declarations" section of
// vulkan_renderer.cu (~line 200):
//   static void initBillboardSystem(const SimParams& params);
//   static void updateBillboardVBO(const SnowballState& ball);
//   static void drawBillboards(VkCommandBuffer cmd);
//   static void destroyBillboardSystem(void);
// ============================================================================
// IntelliSense guard: questi include vengono skippati a compile time
// (già presenti in vulkan_renderer.cu), ma VS Code li vede standalone.
#ifndef VK_VERSION_1_0
  #ifdef _WIN32
    #define VK_USE_PLATFORM_WIN32_KHR
    #ifndef NOMINMAX
      #define NOMINMAX
    #endif
    #include <windows.h>
  #endif
  
  #define GLFW_INCLUDE_VULKAN

  #include <GLFW/glfw3.h>
  #include <vulkan/vulkan.h>
  #include <cuda_runtime.h>

  #include <cstdio>
  #include <cstdlib>
  #include <cstring>
  #include <cmath>
  #include <vector>
  #include <string>

  #ifndef _WIN32
    #include <unistd.h>
  #endif

  #include "vulkan_renderer.h"
  #include "stb_image.h"

  #include "include/helpers.cuh"
#endif

// ----------------------------------------------------------------------------
// drawBillboardUI
// Call inside drawCameraSceneUI(), renderFrame() after ImGui::NewFrame()
// ----------------------------------------------------------------------------
static void drawBillboardUI(void)
{
    ImGui::Begin("Billboard Positions");

    ImGui::TextColored(ImVec4(1.f, 0.4f, 0.2f, 1.f), "Angry Santa");
    ImGui::SliderFloat("Santa X", &g_bb.santaPos[0], -50.f, 200.f, "%.1f");
    ImGui::SliderFloat("Santa Y", &g_bb.santaPos[1], -10.f, 100.f, "%.1f");
    ImGui::SliderFloat("Santa Z", &g_bb.santaPos[2], -30.f,  30.f, "%.1f");

    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.4f, 1.f), "Christmas Village");
    ImGui::SliderFloat("Village X", &g_bb.villPos[0], -50.f, 300.f, "%.1f");
    ImGui::SliderFloat("Village Y", &g_bb.villPos[1], -10.f, 100.f, "%.1f");
    ImGui::SliderFloat("Village Z", &g_bb.villPos[2], -30.f,  30.f, "%.1f");

    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 1.f, 1.f), "Sizes");
    ImGui::SliderFloat("Santa W",  &g_bb.santaW, 0.1f, 200.f, "%.1f");
    ImGui::SliderFloat("Santa H",  &g_bb.santaH, 0.1f, 200.f, "%.1f");
    ImGui::SliderFloat("Village W",&g_bb.villW,  1.f,  400.f, "%.1f");
    ImGui::SliderFloat("Village H",&g_bb.villH,  1.f,  400.f, "%.1f");

    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.f, 1.f, 0.3f, 1.f), "Chroma Key");
    ImGui::SliderFloat("Santa Thresh",  &g_bb.ck[0][3], 0.f, 0.8f, "%.3f");
    ImGui::SliderFloat("Village Thresh",&g_bb.ck[1][3], 0.f, 0.8f, "%.3f");

    ImGui::Spacing();
    if (ImGui::Button("Print values")) {
        printf("[Billboard] Santa  pos=(%.1f, %.1f, %.1f) size=(%.1f x %.1f)\n",
               g_bb.santaPos[0], g_bb.santaPos[1], g_bb.santaPos[2],
               g_bb.santaW, g_bb.santaH);
        printf("[Billboard] Village pos=(%.1f, %.1f, %.1f) size=(%.1f x %.1f)\n",
               g_bb.villPos[0], g_bb.villPos[1], g_bb.villPos[2],
               g_bb.villW, g_bb.villH);
        printf("[Billboard] Chroma  santa=%.3f  village=%.3f\n",
               g_bb.ck[0][3], g_bb.ck[1][3]);
    }

    ImGui::End();
}

// ============================================================================
// Internal helpers
// ============================================================================

// Single-time command buffer: begin
static VkCommandBuffer bbBeginCmd(void)
{
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = g_ctx.commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(g_ctx.device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

// Single-time command buffer: end + submit + wait
static void bbEndCmd(VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(g_ctx.graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(g_ctx.graphicsQueue);
    vkFreeCommandBuffers(g_ctx.device, g_ctx.commandPool, 1, &cmd);
}

// Image layout transition
static void bbTransition(VkCommandBuffer cmd, VkImage img,
                          VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout           = from;
    b.newLayout           = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = img;
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkPipelineStageFlags srcStage, dstStage;
    if (from == VK_IMAGE_LAYOUT_UNDEFINED) {
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else {   // TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0,
                         0, nullptr, 0, nullptr, 1, &b);
}

// Load PNG → VkImage + VkDeviceMemory + VkImageView at g_bb slot
static void bbLoadTexture(const char* path, int slot)
{
    uint8_t* pixelData = nullptr;
    bool fallback = false;
    
    int fileW = 0, fileH = 0, fileChannels = 0;
    pixelData = stbi_load(path, &fileW, &fileH, &fileChannels, 4);
    if (!pixelData) {
      printf("[Billboard] Failed to load texture: %s\n", stbi_failure_reason());
      fallback = true;
    }

    VkDeviceSize sz = (VkDeviceSize)fileW * fileH * 4;

    // ---- Staging buffer ----
    VkBuffer stgBuf; VkDeviceMemory stgMem;
    {
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = sz;
        bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(g_ctx.device, &bci, nullptr, &stgBuf);

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(g_ctx.device, stgBuf, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(g_ctx.device, &ai, nullptr, &stgMem);
        vkBindBufferMemory(g_ctx.device, stgBuf, stgMem, 0);

        void* d;
        vkMapMemory(g_ctx.device, stgMem, 0, sz, 0, &d);
        memcpy(d, pixelData, sz);
        vkUnmapMemory(g_ctx.device, stgMem);
    }

    if (fallback) free(pixelData); else stbi_image_free(pixelData);

    // ---- VkImage ----
    {
        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = VK_FORMAT_R8G8B8A8_SRGB;
        ici.extent        = {(uint32_t)fileW, (uint32_t)fileH, 1};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCreateImage(g_ctx.device, &ici, nullptr, &g_bb.img[slot]);

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(g_ctx.device, g_bb.img[slot], &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(g_ctx.device, &ai, nullptr, &g_bb.mem[slot]);
        vkBindImageMemory(g_ctx.device, g_bb.img[slot], g_bb.mem[slot], 0);
    }

    // ---- Upload via staging ----
    {
        VkCommandBuffer cmd = bbBeginCmd();
        bbTransition(cmd, g_bb.img[slot],
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent      = {(uint32_t)fileW, (uint32_t)fileH, 1};
        vkCmdCopyBufferToImage(cmd, stgBuf, g_bb.img[slot],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        bbTransition(cmd, g_bb.img[slot],
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        bbEndCmd(cmd);
    }

    vkDestroyBuffer(g_ctx.device, stgBuf, nullptr);
    vkFreeMemory(g_ctx.device, stgMem, nullptr);

    // ---- Image view ----
    VkImageViewCreateInfo vci{};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = g_bb.img[slot];
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = VK_FORMAT_R8G8B8A8_SRGB;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(g_ctx.device, &vci, nullptr, &g_bb.view[slot]);

    printf("[Billboard] Slot %d loaded: '%s' (%dx%d)\n", slot, path, fileW, fileH);
}

// Write/update one descriptor set to point at the image in slot
static void bbWriteDescSet(int slot)
{
    VkDescriptorImageInfo ii{};
    ii.sampler     = g_bb.sampler;
    ii.imageView   = g_bb.view[slot];
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet wr{};
    wr.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr.dstSet          = g_bb.ds[slot];
    wr.dstBinding      = 0;
    wr.descriptorCount = 1;
    wr.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr.pImageInfo      = &ii;
    vkUpdateDescriptorSets(g_ctx.device, 1, &wr, 0, nullptr);
}

// Build 6 vertices (2 triangles) for one billboard quad into verts[voff..voff+5]
static void bbBuildQuad(BillboardVertex* verts, int voff,
                         float cx, float cy, float cz,
                         float w,  float h,
                         float rx, float ry, float rz,   // camera-right (normalised)
                         float ux, float uy, float uz)   // world-up (0,1,0)
{
    float hw = w * 0.5f, hh = h * 0.5f;
    // 4 corners
    float TL[3] = { cx - rx*hw + ux*hh,  cy - ry*hw + uy*hh,  cz - rz*hw + uz*hh };
    float TR[3] = { cx + rx*hw + ux*hh,  cy + ry*hw + uy*hh,  cz + rz*hw + uz*hh };
    float BL[3] = { cx - rx*hw - ux*hh,  cy - ry*hw - uy*hh,  cz - rz*hw - uz*hh };
    float BR[3] = { cx + rx*hw - ux*hh,  cy + ry*hw - uy*hh,  cz + rz*hw - uz*hh };

    // Two CCW triangles (Vulkan default winding, viewed from camera)
    BillboardVertex* v = verts + voff;
    v[0] = { TL[0], TL[1], TL[2],  0.f, 0.f };
    v[1] = { BL[0], BL[1], BL[2],  0.f, 1.f };
    v[2] = { BR[0], BR[1], BR[2],  1.f, 1.f };
    v[3] = { TL[0], TL[1], TL[2],  0.f, 0.f };
    v[4] = { BR[0], BR[1], BR[2],  1.f, 1.f };
    v[5] = { TR[0], TR[1], TR[2],  1.f, 0.f };
}


// ============================================================================
// Public API
// ============================================================================

// ----------------------------------------------------------------------------
// initBillboardSystem
// Call at the end of initRenderer(), after createSlopePipeline()
// ----------------------------------------------------------------------------
static void initBillboardSystem(const SimParams& params)
{
    // ---- Billboard world positions ----
    // Santa: near the top of the slope (slopeTopY = H/cos)
    g_bb.santaPos[0] = 2.0f;
    g_bb.santaPos[1] = params.slopeHeight / params.slopeCos + 0.5f;
    g_bb.santaPos[2] = 0.0f;

    // Village: just past where slope meets ground (groundStartX = H/sin)
    float groundStartX = params.slopeHeight / params.slopeSin;
    g_bb.villPos[0] = groundStartX + 1.0f;
    g_bb.villPos[1] = 1.5f;
    g_bb.villPos[2] = -0.9f;
    g_bb.destroyX = g_bb.villPos[0];

    // ---- Load textures ----
    // EXEPATH is defined in CMakeLists
    const char* path1 = EXEPATH "textures/Angry_Santa.png";    
    bbLoadTexture(path1, 0);
    const char* path2 = EXEPATH "textures/Christmas_Village.png";
    bbLoadTexture(path2, 1);
    const char* path3 = EXEPATH "textures/Christmas_Village_Destroyed.png";
    bbLoadTexture(path3, 2);

    // ---- Sampler (linear, clamp to edge) ----
    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(g_ctx.device, &sci, nullptr, &g_bb.sampler);

    // ---- Descriptor set layout (binding 0 = combined image sampler) ----
    VkDescriptorSetLayoutBinding dslb{};
    dslb.binding         = 0;
    dslb.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    dslb.descriptorCount = 1;
    dslb.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 1;
    dlci.pBindings    = &dslb;
    vkCreateDescriptorSetLayout(g_ctx.device, &dlci, nullptr, &g_bb.descLayout);

    // ---- Descriptor pool (3 sets) ----
    VkDescriptorPoolSize ps{};
    ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps.descriptorCount = 3;

    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 3;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &ps;
    vkCreateDescriptorPool(g_ctx.device, &dpci, nullptr, &g_bb.descPool);

    // ---- Descriptor sets ----
    VkDescriptorSetLayout layouts[3] = {g_bb.descLayout, g_bb.descLayout, g_bb.descLayout};
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = g_bb.descPool;
    dsai.descriptorSetCount = 3;
    dsai.pSetLayouts        = layouts;
    vkAllocateDescriptorSets(g_ctx.device, &dsai, g_bb.ds);
    for (int i = 0; i < 3; i++) bbWriteDescSet(i);

    // ---- Pipeline layout (descriptor + push constants) ----
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(BillboardPC);   // 80 bytes

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &g_bb.descLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    vkCreatePipelineLayout(g_ctx.device, &plci, nullptr, &g_bb.pipelineLayout);

    // ---- Graphics pipeline ----
    auto vertCode = readSPIRV("shaders/billboard_vert.spv");
    auto fragCode = readSPIRV("shaders/billboard_frag.spv");
    VkShaderModule vertMod = createShaderModule(vertCode);
    VkShaderModule fragMod = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName  = "main";

    // Vertex input: vec3 pos + vec2 uv
    VkVertexInputBindingDescription vib{};
    vib.binding   = 0;
    vib.stride    = sizeof(BillboardVertex);
    vib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription via[2]{};
    via[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(BillboardVertex, x)};
    via[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT,    (uint32_t)offsetof(BillboardVertex, u)};

    VkPipelineVertexInputStateCreateInfo viState{};
    viState.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    viState.vertexBindingDescriptionCount   = 1;
    viState.pVertexBindingDescriptions      = &vib;
    viState.vertexAttributeDescriptionCount = 2;
    viState.pVertexAttributeDescriptions    = via;

    VkPipelineInputAssemblyStateCreateInfo iaState{};
    iaState.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.width    = (float)g_ctx.swapchainExtent.width;
    viewport.height   = (float)g_ctx.swapchainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{};
    scissor.extent = g_ctx.swapchainExtent;

    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1;
    vpState.pViewports    = &viewport;
    vpState.scissorCount  = 1;
    vpState.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rsState{};
    rsState.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rsState.polygonMode = VK_POLYGON_MODE_FILL;
    rsState.cullMode    = VK_CULL_MODE_NONE;   // no culling - always visible
    rsState.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsState.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo msState{};
    msState.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth test ON, depth write OFF (transparent billboard, doesn't occlude)
    VkPipelineDepthStencilStateCreateInfo dsState{};
    dsState.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dsState.depthTestEnable  = VK_TRUE;
    dsState.depthWriteEnable = VK_FALSE;
    dsState.depthCompareOp   = VK_COMPARE_OP_LESS;

    // Standard alpha blending
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable         = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp        = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blend.alphaBlendOp        = VK_BLEND_OP_ADD;
    blend.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cbState{};
    cbState.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbState.attachmentCount = 1;
    cbState.pAttachments    = &blend;

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount          = 2;
    gpci.pStages             = stages;
    gpci.pVertexInputState   = &viState;
    gpci.pInputAssemblyState = &iaState;
    gpci.pViewportState      = &vpState;
    gpci.pRasterizationState = &rsState;
    gpci.pMultisampleState   = &msState;
    gpci.pDepthStencilState  = &dsState;
    gpci.pColorBlendState    = &cbState;
    gpci.layout              = g_bb.pipelineLayout;
    gpci.renderPass          = g_ctx.renderPass;
    gpci.subpass             = 0;

    VkResult r = vkCreateGraphicsPipelines(g_ctx.device, VK_NULL_HANDLE, 1, &gpci,
                                           nullptr, &g_bb.pipeline);
    if (r != VK_SUCCESS)
        fprintf(stderr, "[Billboard] Pipeline creation FAILED: %d\n", (int)r);
    else
        printf("[Billboard] Pipeline created\n");

    vkDestroyShaderModule(g_ctx.device, vertMod, nullptr);
    vkDestroyShaderModule(g_ctx.device, fragMod, nullptr);

    // ---- Host-visible VBO (stays mapped permanently) ----
    {
        VkDeviceSize vboSz = 12 * sizeof(BillboardVertex);
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = vboSz;
        bci.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(g_ctx.device, &bci, nullptr, &g_bb.vbo);

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(g_ctx.device, g_bb.vbo, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(g_ctx.device, &ai, nullptr, &g_bb.vboMem);
        vkBindBufferMemory(g_ctx.device, g_bb.vbo, g_bb.vboMem, 0);
        vkMapMemory(g_ctx.device, g_bb.vboMem, 0, vboSz, 0, &g_bb.mapped);
    }

    printf("[Billboard] System ready – Santa (%.1f,%.1f,%.1f)  Village (%.1f,%.1f,%.1f)\n",
           g_bb.santaPos[0], g_bb.santaPos[1], g_bb.santaPos[2],
           g_bb.villPos[0],  g_bb.villPos[1],  g_bb.villPos[2]);
}


// ----------------------------------------------------------------------------
// resetBillboardState
// Call on each restart (from notifySimStarted) to reset the village to
// non-destroyed state and recalculate positions from the (possibly new) params.
// ----------------------------------------------------------------------------
static void resetBillboardState(const SimParams& params)
{
    g_bb.destroyed = false;

    // Recompute village X to stay near the slope-ground junction
    float groundStartX = params.slopeHeight / params.slopeSin;
    g_bb.villPos[0] = groundStartX + 1.0f;
    g_bb.destroyX   = g_bb.villPos[0];

    // Keep Santa near the slope top
    g_bb.santaPos[1] = params.slopeHeight / params.slopeCos + 0.5f;

    printf("[Billboard] State reset – village X=%.1f (groundStart=%.1f), Santa Y=%.1f\n",
           g_bb.villPos[0], groundStartX, g_bb.santaPos[1]);
}


// ----------------------------------------------------------------------------
// updateBillboardVBO
// Call in renderFrame(), after the VBO fill section and before recordCommandBuffer
// ----------------------------------------------------------------------------
static void updateBillboardVBO(const SnowballState& ball)
{
    // Check village destruction: trigger when the ball's leading surface reaches the village,
    // not just the center (accounts for the dynamically growing ball radius).
    if (!g_bb.destroyed && (ball.posX + ball.radius) >= g_bb.destroyX) {
        g_bb.destroyed = true;
        printf("[Billboard] Village destroyed! (ball.posX=%.1f + radius=%.2f >= destroyX=%.1f)\n",
               ball.posX, ball.radius, g_bb.destroyX);
    }

    // Derive camera-right vector from spherical angles stored in g_ctx
    float cosP = cosf(g_ctx.camPhi);
    float sinT = sinf(g_ctx.camTheta);
    float cosT = cosf(g_ctx.camTheta);

    // Forward direction (world space, unnormalised but dist cancels)
    float fx = cosP * sinT;
    float fz = cosP * cosT;

    // right = forward × world_up(0,1,0) = (fz, 0, -fx), then normalise
    float rx = fz, ry = 0.f, rz = -fx;
    float rlen = sqrtf(rx*rx + rz*rz);
    if (rlen > 1e-6f) { rx /= rlen; rz /= rlen; }

    float ux = 0.f, uy = 1.f, uz = 0.f;  // world up

    BillboardVertex verts[12];
    bbBuildQuad(verts, 0,
        g_bb.santaPos[0], g_bb.santaPos[1], g_bb.santaPos[2],
        g_bb.santaW, g_bb.santaH,
        rx, ry, rz, ux, uy, uz);
    bbBuildQuad(verts, 6,
        g_bb.villPos[0], g_bb.villPos[1], g_bb.villPos[2],
        g_bb.villW, g_bb.villH,
        rx, ry, rz, ux, uy, uz);

    memcpy(g_bb.mapped, verts, sizeof(verts));
}


// ----------------------------------------------------------------------------
// drawBillboards
// Call inside recordCommandBuffer(), after "Draw 3: Particles" and before ImGui
// ----------------------------------------------------------------------------
static void drawBillboards(VkCommandBuffer cmd)
{
    if (!g_bb.pipeline || !g_bb.vbo) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_bb.pipeline);

    BillboardPC pc{};
    memcpy(pc.mvp, g_ctx.mvp, sizeof(float)*16);

    // ---- Draw Santa (vertices 0-5) ----
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &g_bb.vbo, &offset);

    pc.chromaR = g_bb.ck[0][0];  pc.chromaG = g_bb.ck[0][1];
    pc.chromaB = g_bb.ck[0][2];  pc.chromaThresh = g_bb.ck[0][3];
    vkCmdPushConstants(cmd, g_bb.pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(BillboardPC), &pc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        g_bb.pipelineLayout, 0, 1, &g_bb.ds[0], 0, nullptr);
    vkCmdDraw(cmd, 6, 1, 0, 0);

    // ---- Draw Village (vertices 6-11) – normal or destroyed ----
    offset = 6 * sizeof(BillboardVertex);
    vkCmdBindVertexBuffers(cmd, 0, 1, &g_bb.vbo, &offset);

    pc.chromaR = g_bb.ck[1][0];  pc.chromaG = g_bb.ck[1][1];
    pc.chromaB = g_bb.ck[1][2];  pc.chromaThresh = g_bb.ck[1][3];
    vkCmdPushConstants(cmd, g_bb.pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(BillboardPC), &pc);
    int villSlot = g_bb.destroyed ? 2 : 1;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        g_bb.pipelineLayout, 0, 1, &g_bb.ds[villSlot], 0, nullptr);
    vkCmdDraw(cmd, 6, 1, 0, 0);
}


// ----------------------------------------------------------------------------
// destroyBillboardSystem
// Call at the beginning of destroyRenderer(), before other Vulkan teardown
// ----------------------------------------------------------------------------
static void destroyBillboardSystem(void)
{
    if (!g_ctx.device) return;
    vkDeviceWaitIdle(g_ctx.device);

    if (g_bb.pipeline)       vkDestroyPipeline(g_ctx.device, g_bb.pipeline, nullptr);
    if (g_bb.pipelineLayout) vkDestroyPipelineLayout(g_ctx.device, g_bb.pipelineLayout, nullptr);
    if (g_bb.descPool)       vkDestroyDescriptorPool(g_ctx.device, g_bb.descPool, nullptr);
    if (g_bb.descLayout)     vkDestroyDescriptorSetLayout(g_ctx.device, g_bb.descLayout, nullptr);
    if (g_bb.sampler)        vkDestroySampler(g_ctx.device, g_bb.sampler, nullptr);

    for (int i = 0; i < 3; i++) {
        if (g_bb.view[i]) vkDestroyImageView(g_ctx.device, g_bb.view[i], nullptr);
        if (g_bb.img[i])  vkDestroyImage(g_ctx.device, g_bb.img[i], nullptr);
        if (g_bb.mem[i])  vkFreeMemory(g_ctx.device, g_bb.mem[i], nullptr);
    }

    if (g_bb.vboMem) vkFreeMemory(g_ctx.device, g_bb.vboMem, nullptr);
    if (g_bb.vbo)    vkDestroyBuffer(g_ctx.device, g_bb.vbo, nullptr);

    printf("[Billboard] System destroyed\n");
}
