################################################################################
# Copyright (c) 2023, Your Name. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#################################################################################

CUDA_VER?=11.4
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

TARGET_DEVICE = $(shell gcc -g -dumpmachine | cut -f1 -d -)
CXX:= g++
SRCS:= gstnvsegcustomvisual.cpp
INCS:= $(wildcard *.h)
LIB:= libnvdsgst_customnvsegvisual.so

NVDS_VERSION:=6.3

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

CFLAGS+= -fPIC -DDS_VERSION=\"6.3.0\" -g \
	-I /usr/local/cuda-$(CUDA_VER)/include \
	-I ../../includes \
	-I /usr/include/gstreamer-1.0 \
	-I /usr/include/orc-0.4 \
	-I /usr/include/glib-2.0 \
	-I /usr/lib/aarch64-linux-gnu/glib-2.0/include \
	-I /usr/src/jetson_multimedia_api/include \
	-I /opt/nvidia/deepstream/deepstream-6.3/include -I /opt/nvidia/deepstream/deepstream/sources/includes

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
	-lnppc -lnppig -lnpps -lnppicc -lnppidei \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvdsbufferpool -lnvbufsurftransform \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)

ifeq ($(TARGET_DEVICE),aarch64)
	PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 opencv4
else
	PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 opencv4s
endif

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)

