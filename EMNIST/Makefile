CXX = g++
CXXFLAGS = -Wall -g
INCLUDES = -I/usr/local/include/opencv4 -I./include
LFLAGS = -L/usr/local/lib
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_xfeatures2d -lopencv_videoio -lopencv_ml
SRCS = src/main.cpp include/Storage.cpp include/PCA.cpp include/SVM.cpp
OBJS = $(SRCS:.cpp=.o)
MAIN = main

.PHONY: clean

# Target rules here
all: $(MAIN)
	@echo Simple compiler named EMMNIST TEST has been compiled

$(MAIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)
	@$(RM) $(OBJS)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)