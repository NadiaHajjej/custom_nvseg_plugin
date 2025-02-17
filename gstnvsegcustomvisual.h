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

#ifndef __GST_CUSTOM_NVSEG_VISUAL_H__
#define __GST_CUSTOM_NVSEG_VISUAL_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

/* Package and library details required for plugin_init */
#define PACKAGE "customnvsegvisual"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA Custom segmentation visualization plugin for integration with DeepStream on DGPU"
#define BINARY_PACKAGE "NVIDIA Custom DeepStream Segmentation Visualization Plugin"
#define URL "http://nvidia.com/"


G_BEGIN_DECLS
/* Standard boilerplate stuff */
typedef struct _GstCustomNvSegVisual GstCustomNvSegVisual;
typedef struct _GstCustomNvSegVisualClass GstCustomNvSegVisualClass;

/* Standard boilerplate stuff */
#define GST_TYPE_CUSTOM_NV_SEG_VISUAL (gst_custom_nvseg_visual_get_type())
#define GST_CUSTOM_NV_SEG_VISUAL(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_CUSTOM_NV_SEG_VISUAL,GstCustomNvSegVisual))
#define GST_CUSTOM_NV_SEG_VISUAL_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_CUSTOM_NV_SEG_VISUAL,GstCustomNvSegVisualClass))
#define GST_CUSTOM_NV_SEG_VISUAL_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_CUSTOM_NV_SEG_VISUAL, GstCustomNvSegVisualClass))
#define GST_IS_CUSTOM_NV_SEG_VISUAL(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_CUSTOM_NV_SEG_VISUAL))
#define GST_IS_CUSTOM_NV_SEG_VISUAL_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_CUSTOM_NV_SEG_VISUAL))
#define GST_CUSTOM_NV_SEG_VISUAL_CAST(obj)  ((GstCustomNvSegVisual *)(obj))

struct _GstCustomNvSegVisual
{
  GstBaseTransform base_trans;

  // Unique ID of the element. The labels generated by the element will be
  // updated at index `unique_id` of attr_info array in NvDsObjectParams.
  guint unique_id;

  // Frame number of the current input buffer
  guint64 frame_num;

  // Input video info (resolution, color format, framerate, etc)
  GstVideoInfo video_info;

  // Resolution at which frames/objects should be processed
  gint output_width;
  gint output_height;

  // GPU ID on which we expect to execute the task
  guint gpu_id;

  // Buffer pool size used for optical flow output
  guint pool_size;

  gint batch_size;
  guint num_batch_buffers;

  gint input_feature;
  gint output_feature;
  gint cuda_mem_type;
  GstVideoFormat input_fmt;
  GstVideoFormat output_fmt;

  GstVideoInfo in_info;
  GstVideoInfo out_info;

  GstCaps *sinkcaps;
  GstCaps *srccaps;

  GstBufferPool *pool;
};

// Boiler plate stuff
struct _GstCustomNvSegVisualClass
{
  GstBaseTransformClass parent_class;
};

GType gst_custom_nvseg_visual_get_type (void);

G_END_DECLS
#endif /* __GST_CUSTOM_NVSEG_VISUAL_H__ */

