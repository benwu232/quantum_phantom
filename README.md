
# About Quantum Phantom

QP is a concept prototype of a new technology, which turns a webcam into a non-touch pointing device that
can directly control the objects on screen, and even draw or write on it. It is based on computer vision,
machine learning and pattern recognition.

The demo videos are here:

http://www.youtube.com/watch?v=LUyql0SVobc

http://www.youtube.com/watch?v=ExE5m6BjnV0

or here:

http://v.youku.com/v_show/id_XMjkyMzg5NzQ4.html

http://v.youku.com/v_show/id_XMjk5NTg5MTQ0.html

and the relevant reports are here:

http://www.engadget.com/2011/08/12/quantum-phantom-prototype-lets-you-control-your-computer-screen/

http://www.geek.com/articles/gadgets/quantum-phantom-styles-itself-after-tony-starks-computer-20110812/

## Note:
Because I only trained the classifer for black arrow, you need to set the pointer of the mouse to black arrow to try this program. You can set it on the settings of Windows or Linux.

## Compile & run

### Under Windows
The repository is for Windows, which contains source code and project files. It uses Visual Studio Express 2010 under Windows 7 and should be ok on other Windows and Visual Studios.

There are some steps to compile and run this code:

1. Download and install [OpenCV](http://opencv.org/).
2. (Optional) If you want high performance, open TBB(Thread Building Block). This can make the program use multi-core to compute and run much faster. I'm not sure if the downloaded version has this option turned on. I suggest you compile OpenCV yourself and turn on the TBB option. You can find the detail on OpenCV website.
3. Add the include and library paths of OpenCV to the project. I have done this in the project, please check it for reference.
4. Build the project to release version. If you are lucky, you will be successful and get qp.exe.
5. Goto script directory and run qp_black_arrow.bat. If everything is right. You can do the same things I have done in the demo videos.

### Under Linux
The steps are similar to those in Windows. The program should automatically recognize the OS. If not, please comment the WIN32 parts and release the linux parts.

# Have fun!

