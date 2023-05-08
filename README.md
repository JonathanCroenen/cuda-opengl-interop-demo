# cuda-opengl-interop-demo
This repository contains two CUDA-OpenGL interoperability demos written in Python. 
* The first [demo](https://github.com/JonathanCroenen/cuda-opengl-interop-demo/blob/main/src/demo_egl.py) shows an offscreen rendering scenario using EGL for OpenGL context creation.
* The second [demo](https://github.com/JonathanCroenen/cuda-opengl-interop-demo/blob/main/src/demo_glfw.py) shows a realtime application with GLfW for window and OpenGL context creation.
Both demos show how to map an OpenGL renderbuffer/texture into CUDA so that it can be modified by CUDA kernels. After which the modified result is transferred back so that the changes can be seen in OpenGL. The second demo then displays the modified result in real time.
