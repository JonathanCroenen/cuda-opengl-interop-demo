# cuda-opengl-interop-demo
This repository contains two CUDA-OpenGL interoperability demos written in Python. 
* The first [demo](https://github.com/JonathanCroenen/cuda-opengl-interop-demo/blob/main/src/demo_egl.py) shows an offscreen rendering scenario using EGL for OpenGL context creation.
* The second [demo](https://github.com/JonathanCroenen/cuda-opengl-interop-demo/blob/main/src/demo_glfw.py) shows a realtime application with GLFW for window and OpenGL context creation.

Both demos show how to map an OpenGL renderbuffer/texture into CUDA so that it can be modified by CUDA kernels. After which the modified result is transferred back, so that the changes can be seen in OpenGL. The offscreen demo shows an image plot of each major step in the process after completing and the windowed demo displays the modified result in the application window.

The packages used to interact with OpenGL/EGL and CUDA from Python are [PyOpenGL](https://github.com/mcfletch/pyopengl) and [cuda-python](https://github.com/NVIDIA/cuda-python) respectively. Additionaly, [GLFW bindings](https://github.com/FlorianRhiem/pyGLFW) are used for window and context creation in the second demo.
