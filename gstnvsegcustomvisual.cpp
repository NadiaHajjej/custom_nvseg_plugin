/**
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <stdio.h>        // For FILE operations
#include <stdlib.h>      // For malloc and free
#include <string.h>      // For memcpy
#include <cuda_runtime.h> // Include CUDA runtime for device operations

#include "gstnvdsbufferpool.h"
#include "gstnvsegcustomvisual.h"
#include "nvbufsurface.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"

GST_DEBUG_CATEGORY_STATIC (gst_custom_nvseg_visual_debug);
#define GST_CAT_DEFAULT gst_custom_nvseg_visual_debug

static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
    PROP_0,
    PROP_UNIQUE_ID,
    PROP_GPU_DEVICE_ID,
    PROP_BATCH_SIZE,
    PROP_WIDTH,
    PROP_HEIGHT,
};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 0
#define DEFAULT_OUTPUT_WIDTH 1280
#define DEFAULT_OUTPUT_HEIGHT 720
#define DEFAULT_GPU_ID 0
#define DEFAULT_GRID_SIZE 0

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_custom_nvseg_visual_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                            "memory:NVMM",
                            "{ NV12, RGBA }")));

static GstStaticPadTemplate gst_custom_nvseg_visual_src_template =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                            "memory:NVMM",
                            "{ RGBA }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_custom_nvseg_visual_parent_class parent_class
G_DEFINE_TYPE (GstCustomNvSegVisual, gst_custom_nvseg_visual, GST_TYPE_BASE_TRANSFORM);

static void gst_custom_nvseg_visual_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_custom_nvseg_visual_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_custom_nvseg_visual_transform_size(GstBaseTransform* btrans,
        GstPadDirection dir, GstCaps *caps, gsize size, GstCaps* othercaps, gsize* othersize);

static GstCaps* gst_custom_nvseg_visual_fixate_caps(GstBaseTransform* btrans,
        GstPadDirection direction, GstCaps* caps, GstCaps* othercaps);

static gboolean gst_custom_nvseg_visual_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);

static GstCaps* gst_custom_nvseg_visual_transform_caps(GstBaseTransform* btrans, GstPadDirection dir,
    GstCaps* caps, GstCaps* filter);

static gboolean gst_custom_nvseg_visual_start (GstBaseTransform * btrans);
static gboolean gst_custom_nvseg_visual_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_custom_nvseg_visual_transform(GstBaseTransform* btrans,
    GstBuffer* inbuf, GstBuffer* outbuf);

static GstFlowReturn
gst_custom_nvseg_visual_prepare_output_buffer (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer ** outbuf);


/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_custom_nvseg_visual_class_init (GstCustomNvSegVisualClass * klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;
    GstBaseTransformClass *gstbasetransform_class;
    gobject_class = (GObjectClass *) klass;
    gstelement_class = (GstElementClass *) klass;
    gstbasetransform_class = (GstBaseTransformClass *) klass;

    /* Overide base class functions */
    gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_custom_nvseg_visual_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_custom_nvseg_visual_get_property);

    gstbasetransform_class->transform_size = GST_DEBUG_FUNCPTR(gst_custom_nvseg_visual_transform_size);
    gstbasetransform_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_custom_nvseg_visual_fixate_caps);
    gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_custom_nvseg_visual_set_caps);
    gstbasetransform_class->transform_caps = GST_DEBUG_FUNCPTR(gst_custom_nvseg_visual_transform_caps);
    gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_custom_nvseg_visual_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_custom_nvseg_visual_stop);

    gstbasetransform_class->transform = GST_DEBUG_FUNCPTR (gst_custom_nvseg_visual_transform);

    gstbasetransform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR (gst_custom_nvseg_visual_prepare_output_buffer);

    gstbasetransform_class->passthrough_on_same_caps = TRUE;

    /* Install properties */
    g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint ("unique-id",
            "Unique ID",
            "Unique ID for the element. Can be used to identify output of the"
            " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
            (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
        g_param_spec_uint ("gpu-id",
            "Set GPU Device ID",
            "Set GPU Device ID", 0,
            G_MAXUINT, 0,
            GParamFlags
            (G_PARAM_READWRITE |
                G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
        g_param_spec_uint ("batch-size", "Batch Size",
            "Maximum batch size for inference",
            1, G_MAXUINT, 1,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property (gobject_class, PROP_WIDTH,
        g_param_spec_uint ("width", "Width",
            "Width of each frame in output batched buffer.",
            0, G_MAXUINT, DEFAULT_OUTPUT_WIDTH,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_HEIGHT,
        g_param_spec_uint ("height", "Height",
            "Height of each frame in output batched buffer.",
            0, G_MAXUINT, DEFAULT_OUTPUT_HEIGHT,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    /* Set sink and src pad capabilities */
    gst_element_class_add_pad_template (gstelement_class,
        gst_static_pad_template_get (&gst_custom_nvseg_visual_src_template));
    gst_element_class_add_pad_template (gstelement_class,
        gst_static_pad_template_get (&gst_custom_nvseg_visual_sink_template));

    /* Set metadata describing the element */
    gst_element_class_set_details_simple(gstelement_class,
          "customnvsegVisual",
          "customnvsegVisual",
          "Gstreamer NV Segmantation Visualization Plugin",
          "NVIDIA Corporation. Post on Deepstream for Jetson/Tesla forum for any queries "
          "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_custom_nvseg_visual_init (GstCustomNvSegVisual * segvisual)
{
    segvisual->sinkcaps =
      gst_static_pad_template_get_caps (&gst_custom_nvseg_visual_sink_template);
    segvisual->srccaps =
      gst_static_pad_template_get_caps (&gst_custom_nvseg_visual_src_template);

    /* Initialize all property variables to default values */
    segvisual->unique_id = DEFAULT_UNIQUE_ID;
    segvisual->output_width = DEFAULT_OUTPUT_WIDTH;
    segvisual->output_height = DEFAULT_OUTPUT_HEIGHT;
    segvisual->gpu_id = DEFAULT_GPU_ID;
    segvisual->batch_size = 1;

#if defined(__aarch64__)
    segvisual->cuda_mem_type = NVBUF_MEM_DEFAULT;
#else
    segvisual->cuda_mem_type = NVBUF_MEM_CUDA_UNIFIED;
#endif

    /* This quark is required to identify NvDsMeta when iterating through
     * the buffer metadatas */
    if (!_dsmeta_quark)
      _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_custom_nvseg_visual_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
    GstCustomNvSegVisual *segvisual = GST_CUSTOM_NV_SEG_VISUAL (object);
    switch (prop_id) {
      case PROP_UNIQUE_ID:
        segvisual->unique_id = g_value_get_uint (value);
        break;
      case PROP_GPU_DEVICE_ID:
        segvisual->gpu_id = g_value_get_uint (value);
        break;
      case PROP_BATCH_SIZE:
        segvisual->batch_size = g_value_get_uint (value);
        break;
      case PROP_WIDTH:
        segvisual->output_width = g_value_get_uint (value);
        break;
      case PROP_HEIGHT:
        segvisual->output_height = g_value_get_uint (value);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_custom_nvseg_visual_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
    GstCustomNvSegVisual *segvisual = GST_CUSTOM_NV_SEG_VISUAL (object);
    switch (prop_id) {
      case PROP_UNIQUE_ID:
        g_value_set_uint (value, segvisual->unique_id);
        break;
      case PROP_GPU_DEVICE_ID:
        g_value_set_uint (value, segvisual->gpu_id);
        break;
      case PROP_BATCH_SIZE:
        g_value_set_uint (value, segvisual->batch_size);
        break;
      case PROP_WIDTH:
        g_value_set_uint (value, segvisual->output_width);
        break;
      case PROP_HEIGHT:
        g_value_set_uint (value, segvisual->output_height);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_custom_nvseg_visual_start(GstBaseTransform *btrans)
{
  GstCustomNvSegVisual *segvisual = GST_CUSTOM_NV_SEG_VISUAL (btrans);
  GST_DEBUG_OBJECT (segvisual, "gst_custom_nvseg_visual_start\n");
	return TRUE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_custom_nvseg_visual_stop (GstBaseTransform * btrans)
{
  GstCustomNvSegVisual *segvisual = GST_CUSTOM_NV_SEG_VISUAL (btrans);

  if (segvisual->pool) {
    gst_buffer_pool_set_active (segvisual->pool, FALSE);
    gst_object_unref(segvisual->pool);
    segvisual->pool = NULL;
  }

  GST_DEBUG_OBJECT (segvisual, "gst_custom_nvseg_visual_stop\n");
  return TRUE;
}

static gboolean
gst_custom_nvseg_visual_transform_size(GstBaseTransform* btrans,
        GstPadDirection dir, GstCaps *caps, gsize size, GstCaps* othercaps, gsize* othersize)
{
    gboolean ret = TRUE;
    GstVideoInfo info;

    ret = gst_video_info_from_caps(&info, othercaps);
    if (ret) *othersize = info.size;

    return ret;
}

static GstCaps *
gst_custom_nvseg_visual_transform_caps (GstBaseTransform * btrans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstCapsFeatures *feature = NULL;
  GstCaps *new_caps = NULL;

  if (direction == GST_PAD_SINK)
  {
    new_caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "RGBA",
          "width", GST_TYPE_INT_RANGE, 1, G_MAXINT, "height", GST_TYPE_INT_RANGE, 1,G_MAXINT, NULL);

  }
  else if (direction == GST_PAD_SRC)
  {
    new_caps = gst_caps_new_simple ("video/x-raw",
          "width", GST_TYPE_INT_RANGE, 1, G_MAXINT, "height", GST_TYPE_INT_RANGE, 1,G_MAXINT, NULL);
  }

  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (new_caps, 0, feature);

  if(gst_caps_is_fixed (caps))
  {
    GstStructure *fs = gst_caps_get_structure (caps, 0);
    const GValue *fps_value;
    guint i, n = gst_caps_get_size(new_caps);

    fps_value = gst_structure_get_value (fs, "framerate");

    // We cannot change framerate
    for (i = 0; i < n; i++)
    {
      fs = gst_caps_get_structure (new_caps, i);
      gst_structure_set_value (fs, "framerate", fps_value);
    }
  }
  return new_caps;
}

/* fixate the caps on the other side */
static GstCaps* gst_custom_nvseg_visual_fixate_caps(GstBaseTransform* btrans,
    GstPadDirection direction, GstCaps* caps, GstCaps* othercaps)
{
  GstCustomNvSegVisual* segvisual = GST_CUSTOM_NV_SEG_VISUAL(btrans);
  GstStructure *s2;
  GstCaps* result;

  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable(othercaps);
  s2 = gst_caps_get_structure(othercaps, 0);

  {
    /* otherwise the dimension of the output heatmap needs to be fixated */
    gst_structure_fixate_field_nearest_int(s2, "width", segvisual->output_width);
    gst_structure_fixate_field_nearest_int(s2, "height", segvisual->output_height);

    gst_structure_remove_fields (s2, "width", "height", NULL);

    gst_structure_set (s2, "width", G_TYPE_INT, segvisual->output_width,
        "height", G_TYPE_INT, segvisual->output_height, NULL);

    result = gst_caps_ref(othercaps);
  }

  gst_caps_unref(othercaps);

  GST_INFO_OBJECT(segvisual, "CAPS fixate: %" GST_PTR_FORMAT ", direction %d",
      result, direction);

  return result;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_custom_nvseg_visual_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstCustomNvSegVisual *segvisual = GST_CUSTOM_NV_SEG_VISUAL (btrans);
  GstStructure *config = NULL;

  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps(&segvisual->video_info, incaps);

  if (segvisual->batch_size == 0)
  {
    g_print ("NvSegVisual: Received invalid batch_size i.e. 0\n");
    return FALSE;
  }

  if (!gst_video_info_from_caps (&segvisual->out_info, outcaps)) {
    GST_ERROR ("invalid output caps");
    return FALSE;
  }
  segvisual->output_fmt = GST_VIDEO_FORMAT_INFO_FORMAT (segvisual->out_info.finfo);

  if (!segvisual->pool)
  {
    segvisual->pool = gst_nvds_buffer_pool_new ();
    config = gst_buffer_pool_get_config (segvisual->pool);

    g_print ("in videoconvert caps = %s\n", gst_caps_to_string(outcaps));
    gst_buffer_pool_config_set_params (config, outcaps, sizeof (NvBufSurface), 4, 4); // TODO: remove 4 hardcoding

    gst_structure_set (config,
        "memtype", G_TYPE_UINT, segvisual->cuda_mem_type,
        "gpu-id", G_TYPE_UINT, segvisual->gpu_id,
        "batch-size", G_TYPE_UINT, segvisual->batch_size, NULL);

    GST_INFO_OBJECT (segvisual, " %s Allocating Buffers in NVM Buffer Pool for Max_Views=%d\n",
        __func__, segvisual->batch_size);

    /* set config for the created buffer pool */
    if (!gst_buffer_pool_set_config (segvisual->pool, config)) {
      GST_WARNING ("bufferpool configuration failed");
      return FALSE;
    }

    gboolean is_active = gst_buffer_pool_set_active (segvisual->pool, TRUE);
    if (!is_active) {
      GST_WARNING (" Failed to allocate the buffers inside the output pool");
      return FALSE;
    } else {
      GST_DEBUG (" Output buffer pool (%p) successfully created",
                  segvisual->pool);
    }
  }

  return TRUE;
}

static GstFlowReturn
gst_custom_nvseg_visual_prepare_output_buffer (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer ** outbuf)
{
  GstBuffer *gstOutBuf = NULL;
  GstFlowReturn result = GST_FLOW_OK;
  GstCustomNvSegVisual *segvisual = GST_CUSTOM_NV_SEG_VISUAL (trans);

  result = gst_buffer_pool_acquire_buffer (segvisual->pool, &gstOutBuf, NULL);
  GST_DEBUG_OBJECT (segvisual, "%s : Frame=%lu Gst-OutBuf=%p\n",
		  __func__, segvisual->frame_num, gstOutBuf);

  if (result != GST_FLOW_OK)
  {
    GST_ERROR_OBJECT (segvisual, "gst_segvisual_prepare_output_buffer failed");
    return result;
  }

  *outbuf = gstOutBuf;
  return result;
}

/* For segmentation visulization */
static unsigned char class2BGR[] = {
  0, 0, 0,        0, 0, 128,      0, 0, 255,
  0, 255, 0,    128, 0, 0,      128, 0, 128,
  128, 128, 0,    0, 128, 0,      0, 0, 64,
  0, 0, 192,      0, 128, 64,     0, 128, 192,
  128, 0, 64,     128, 0, 192,    128, 128, 64,
  128, 128, 192,  0, 64, 0,       0, 64, 128,
  0, 192, 0,     0, 192, 128,    128, 64, 0,
  192, 192, 0
};

void saveNvBufSurfaceToImage(NvBufSurface *surface, const std::string &filePath, int j) {
    // Ensure the surface is valid
    if (!surface || surface->numFilled < 1) {
        std::cerr << "Invalid NvBufSurface" << std::endl;
        return;
    }

    // Map the surface for reading
    if (NvBufSurfaceMap(surface, 0, 0, NVBUF_MAP_READ) != 0) {
        std::cerr << "Failed to map NvBufSurface" << std::endl;
        return;
    }

    // Get image parameters
    int width = surface->surfaceList[j].width;
    int height = surface->surfaceList[j].height;
    int pitch = surface->surfaceList[j].pitch;
    NvBufSurfaceColorFormat format = surface->surfaceList[j].colorFormat;
    g_print("format %d ,or %d, or %d", format, NVBUF_COLOR_FORMAT_NV12_709, NVBUF_COLOR_FORMAT_RGBA);

    // Prepare buffer for the RGBA image
    unsigned char *rgba_image = (unsigned char *)malloc(width * height * 4);  // 4 bytes per pixel for RGBA

    if (format == NVBUF_COLOR_FORMAT_NV12_709 || format == NVBUF_COLOR_FORMAT_NV12_709) {
        // NV12 format: Two planes - Y and UV
        unsigned char *y_plane = (unsigned char *)surface->surfaceList[j].mappedAddr.addr[0];
        unsigned char *uv_plane = (unsigned char *)surface->surfaceList[j].mappedAddr.addr[1];

        // Convert YUV to RGBA
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int y_index = h * pitch + w;
                int uv_index = (h / 2) * (pitch / 2) + (w / 2);

                unsigned char Y = y_plane[y_index];
                unsigned char U = uv_plane[uv_index * 2];
                unsigned char V = uv_plane[uv_index * 2 + 1];

                // Convert YUV to RGB
                int r = Y + 1.402 * (V - 128);
                int g = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128);
                int b = Y + 1.772 * (U - 128);

                // Clamp the values to [0, 255]
                r = std::min(255, std::max(0, r));
                g = std::min(255, std::max(0, g));
                b = std::min(255, std::max(0, b));

                // Fill in the RGBA buffer
                int rgba_index = (h * width + w) * 4;
                rgba_image[rgba_index] = b;        // Blue
                rgba_image[rgba_index + 1] = g;    // Green
                rgba_image[rgba_index + 2] = r;    // Red
                rgba_image[rgba_index + 3] = 255;  // Alpha
            }
        }
    } else if (format == NVBUF_COLOR_FORMAT_RGBA) {
        // RGBA format: Single plane
        unsigned char *data = (unsigned char *)surface->surfaceList[j].mappedAddr.addr[0];
        
        for (int h = 0; h < height; h++) {
            memcpy(rgba_image + h * width * 4, data + h * pitch, width * 4);
        }
    } else {
        std::cerr << "Unsupported format: " << format << std::endl;
        free(rgba_image);
        NvBufSurfaceUnMap(surface, 0, 0);
        return;
    }

    // Create an OpenCV Mat for saving the image
    cv::Mat img(height, width, CV_8UC4, rgba_image);

    // Save the image using OpenCV
    cv::imwrite(filePath, img);

    // Clean up
    free(rgba_image);
    NvBufSurfaceUnMap(surface, 0, 0);
    std::cout << "Saved image to: " << filePath << " (Width: " << width << ", Height: " << height << ")" << std::endl;
}
int c=0;
// Function to overlay color with alpha transparency
static void overlayColorWithAlpha(int* mask, NvBufSurface* orig_buffer, unsigned char* buffer, int height, int width, float alpha=0.3f, int j=0) 
{
  uint8_t *dataPtr = (uint8_t *)orig_buffer->surfaceList[j].mappedAddr.addr[0];
  size_t pitch = orig_buffer->surfaceList[j].pitch;

  g_print("Pitch: %zu\n", pitch);
  g_print("Frame Width: %u\n", width);
  g_print("Frame Height: %u\n", height);
  g_print("Color formar: %d\n", orig_buffer->surfaceList[j].colorFormat);

  unsigned char* buffer_R;
  unsigned char* buffer_G;
  unsigned char* buffer_B;
  unsigned char* buffer_A;
  for (int y = 0; y < height; y++) {
      // Copy each row of pixels from the original image to the buffer
      for (int x = 0; x < width; x++) {
        int  pix_id = (y * width + x);
        // get the image pixel color
        uint8_t *orig_pixel = &dataPtr[y * pitch + x * 4];

        // buffer positon
        buffer_B = buffer + pix_id * 4;
        buffer_G = buffer + pix_id * 4 + 1;
        buffer_R = buffer + pix_id * 4 + 2;
        buffer_A = buffer + pix_id * 4 + 3;

        if (mask[pix_id]<0) {

            *buffer_B = static_cast<unsigned char>(orig_pixel[0]) ;
            *buffer_G = static_cast<unsigned char>(orig_pixel[1]) ;
            *buffer_R = static_cast<unsigned char>(orig_pixel[2])  ;
            *buffer_A = static_cast<unsigned char>(orig_pixel[3])  ;}
        else {
          // for(int pix_id = 0; pix_id < width * height; pix_id++) {
            unsigned char* color = class2BGR + ((mask[pix_id] + 3) * 3);
            // g_print("colors %d,%d,%d,%d",color[0], color[1], color[2], color[3]);
            *buffer_B = static_cast<unsigned char>((orig_pixel[0] * alpha) + (color[0] * (1.0-alpha))) ;
            *buffer_G = static_cast<unsigned char>((orig_pixel[1] * alpha) + (color[1] * (1.0-alpha)));
            *buffer_R = static_cast<unsigned char>((orig_pixel[2] * alpha) + (color[2] * (1.0-alpha))) ;
            *buffer_A = static_cast<unsigned char>((orig_pixel[3] * alpha) + (color[3] * (1.0-alpha))) ;}
       }
  }

#if 0
  // for verif
  if (c <2){
  char file_name[128];
  sprintf(file_name, "sum_map_stream%d_frame%03d.rgba", j, 90);
  FILE* fp = fopen(file_name, "ab");
  fwrite(buffer, 4*height*width, 1, fp);
  fclose(fp);
  c=c+1;
  }
#endif
}


// Transform function to overlay segmentation on the image
static GstFlowReturn
gst_custom_nvseg_visual_transform_internal(GstBaseTransform *btrans,
                                       GstBuffer *inbuf, GstBuffer *outbuf)
{
    GstCustomNvSegVisual *segvisual = GST_CUSTOM_NV_SEG_VISUAL (btrans);
    GstFlowReturn flow_ret = GST_FLOW_OK;
    gpointer state = NULL;
    GstMeta *gst_meta = NULL;
    NvDsMeta *dsmeta = NULL;
    NvDsBatchMeta *batch_meta = NULL;
    guint i = 0;

    GstMapInfo inmap, outmap;

    // Map the input buffer (original image)
    if (!gst_buffer_map(inbuf, &inmap, GST_MAP_READ)) {
        g_print("%s input buffer map failed\n", __func__);
        return GST_FLOW_ERROR;
    }

    // Map the output buffer (where the mask will be overlaid)
    if (!gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE)) {
        g_print("%s output buffer map failed\n", __func__);
        gst_buffer_unmap(inbuf, &inmap);
        return GST_FLOW_ERROR;
    }

    NvBufSurface *dstSurf = (NvBufSurface *)outmap.data;
    NvBufSurface *srcSurf = (NvBufSurface *)inmap.data;

    // Required in the case of tiler
    if (!gst_buffer_copy_into(outbuf, inbuf, GST_BUFFER_COPY_META, 0, -1)) {
        GST_DEBUG("Buffer metadata copy failed \n");
    }

    if (cudaSetDevice(segvisual->gpu_id) != cudaSuccess) {
        g_printerr("Error: failed to set GPU to %d\n", segvisual->gpu_id);
        gst_buffer_unmap(inbuf, &inmap);
        gst_buffer_unmap(outbuf, &outmap);
        return GST_FLOW_ERROR;
    }
    int k=0;
    while ((gst_meta = gst_buffer_iterate_meta(inbuf, &state))) {
        if (gst_meta_api_type_has_tag(gst_meta->info->api, _dsmeta_quark)) {
            dsmeta = (NvDsMeta *) gst_meta;
            if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
                batch_meta = (NvDsBatchMeta *)dsmeta->meta_data;
                break;
            }
        }
    }

    if (batch_meta == NULL) {
        g_print("batch_meta not found, skipping visual execution\n");
        gst_buffer_unmap(inbuf, &inmap);
        gst_buffer_unmap(outbuf, &outmap);
        return GST_FLOW_ERROR;
    }
    
    gst_buffer_unmap(inbuf, &inmap);
    gst_buffer_unmap(outbuf, &outmap);
    dstSurf->numFilled = batch_meta->num_frames_in_batch;
    srcSurf->numFilled = batch_meta->num_frames_in_batch;
  
    g_print("ds_surface %d \n", srcSurf->numFilled);

    if (NvBufSurfaceMap(dstSurf, -1, -1, NVBUF_MAP_WRITE) != 0) {
        g_print("Failed to map destination surface\n");
        return GST_FLOW_ERROR;
    }

    if (NvBufSurfaceMap(srcSurf, -1, -1, NVBUF_MAP_READ) != 0) {
        g_print("Failed to map source surface\n");
        NvBufSurfaceUnMap(dstSurf, -1, -1);
        return GST_FLOW_ERROR;
    }

    GST_DEBUG("Number of frames in source: %u\n", batch_meta->num_frames_in_batch);
    GST_DEBUG("Number of frames in destination: %u\n", batch_meta->num_frames_in_batch);

    static int frame_n = 0;
    for (i = 0; i < batch_meta->num_frames_in_batch; i++) {
        NvDsFrameMeta *frame_meta = nvds_get_nth_frame_meta(batch_meta->frame_meta_list, i);
        if (frame_meta->frame_user_meta_list) {
            NvDsFrameMetaList *fmeta_list = NULL;
            NvDsUserMeta *of_user_meta = NULL;

            for (fmeta_list = frame_meta->frame_user_meta_list; fmeta_list != NULL; fmeta_list = fmeta_list->next) {
                of_user_meta = (NvDsUserMeta *)fmeta_list->data;
                if (of_user_meta && of_user_meta->base_meta.meta_type == NVDSINFER_SEGMENTATION_META) {
                    NvDsInferSegmentationMeta *segmeta = (NvDsInferSegmentationMeta *)(of_user_meta->user_meta_data);
                    GST_DEBUG("classes/width/height=%d/%d/%d\n",
                              segmeta->classes,
                              segmeta->width,
                              segmeta->height);
                    
                    GST_DEBUG("dstSurf [%d] dataSize=%d\n", i, dstSurf->surfaceList[i].dataSize);

                    int rgba_bytes = 4;
                    unsigned char* overlay_buffer = (unsigned char*)(malloc(rgba_bytes * segmeta->height * segmeta->width));
                  

                    overlayColorWithAlpha(segmeta->class_map, srcSurf, overlay_buffer, segmeta->height, segmeta->width, 0.5, i);


                    // // Save the overlay image to disk
                    // g_print("saving_ovrlay");
                    // std::string outputFilePath = "/home/worker/Desktop/custom_nvseg_visual/images/output_overlay_frame_" + std::to_string(frame_n) + ".jpg";
                    // saveNvBufSurfaceToImage(srcSurf, outputFilePath, i);
                    // g_print("format %d\n",dstSurf->surfaceList[i].colorFormat);
#if defined(__aarch64__)
                    // g_print("copy %d"\n, i);
                    for (unsigned int h = 0; h < segmeta->height; h++) {
                        memcpy((char *)dstSurf->surfaceList[i].mappedAddr.addr[0] +
                               h * dstSurf->surfaceList[i].planeParams.pitch[0],
                               overlay_buffer + h * segmeta->width * 4,
                               segmeta->width * 4);
                    }
#else
                    cudaMemcpy((void*)dstSurf->surfaceList[i].mappedAddr.addr[0],
                               (void*)overlay_buffer,
                               rgba_bytes * segmeta->height * segmeta->width,
                               cudaMemcpyHostToDevice);
#endif
                    frame_n++;
                    free(overlay_buffer);
                }
            }
        }
      
      else {
#if defined(__aarch64__)
            for (unsigned int h = 0; h < srcSurf->surfaceList[i].planeParams.height[0]; h++) {
                memcpy((char *)dstSurf->surfaceList[i].mappedAddr.addr[0] +
                       h * dstSurf->surfaceList[i].planeParams.pitch[0],
                       (char *)srcSurf->surfaceList[i].mappedAddr.addr[0] +
                       h * srcSurf->surfaceList[i].planeParams.pitch[0],
                       srcSurf->surfaceList[i].planeParams.width[0] * 4);
            }
#else
            cudaMemcpy((void*)dstSurf->surfaceList[i].mappedAddr.addr[0],
                       (void*)srcSurf->surfaceList[i].mappedAddr.addr[0],
                       srcSurf->surfaceList[i].dataSize,
                       cudaMemcpyDeviceToDevice);
#endif
        }
    

    }


    frame_n++;
    NvBufSurfaceSyncForDevice(dstSurf, -1, -1);
    NvBufSurfaceSyncForDevice(srcSurf, -1, -1);
    NvBufSurfaceUnMap(dstSurf, -1, -1);
    NvBufSurfaceUnMap(srcSurf, -1, -1);

    // gst_buffer_unmap(inbuf, &inmap);
    // gst_buffer_unmap(outbuf, &outmap);

    return flow_ret;
}


/**
 * Called when the plugin works in non-passthough mode
 */
static GstFlowReturn
gst_custom_nvseg_visual_transform(GstBaseTransform* btrans, GstBuffer* inbuf, GstBuffer* outbuf)
{
  return gst_custom_nvseg_visual_transform_internal(btrans, inbuf, outbuf);
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvsegvisualcustom_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_custom_nvseg_visual_debug, "customnvsegvisual", 0,
      "customnvsegvisual plugin");

  return gst_element_register (plugin, "customnvsegvisual", GST_RANK_PRIMARY,
          GST_TYPE_CUSTOM_NV_SEG_VISUAL);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_customnvsegvisual,
    DESCRIPTION, nvsegvisualcustom_plugin_init, "1.0", LICENSE, BINARY_PACKAGE, URL)

