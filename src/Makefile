NVCC     := nvcc
GCC      := gcc
DS_PATH  := /opt/nvidia/deepstream/deepstream

GST_RAW_CFLAGS := $(shell pkg-config --cflags gstreamer-1.0 2>/dev/null || \
	echo "-I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include")
GST_LIBS       := $(shell pkg-config --libs gstreamer-1.0 2>/dev/null || \
	echo "-lgstreamer-1.0 -lgobject-2.0 -lglib-2.0")

DS_INCLUDES := -I$(DS_PATH)/sources/includes
CUDA_LIBS   := -L/usr/local/cuda/lib64 -lcudart

TARGET  := libroi_blur.so
KERNELS := roi_blur_kernels.o
GST_OBJ := roi_blur_gst.o

.PHONY: all clean

all: $(TARGET)

$(KERNELS): roi_blur_kernels.cu
	$(NVCC) -c --compiler-options '-fPIC' -O2 $(DS_INCLUDES) -o $@ $<

$(GST_OBJ): roi_blur_gst.c
	$(GCC) -c -fPIC -O2 $(DS_INCLUDES) $(GST_RAW_CFLAGS) -o $@ $<

$(TARGET): $(KERNELS) $(GST_OBJ)
	$(GCC) -shared -o $@ $^ $(GST_LIBS) $(CUDA_LIBS)

clean:
	rm -f $(TARGET) $(KERNELS) $(GST_OBJ)
