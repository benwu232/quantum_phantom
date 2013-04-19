/*****************************************************************************
* Quantum Phantom
* This is the demo program for the project Quantum Phanotom.
* QP is a concept prototype of a new technology, which turns a webcam into a non-touch pointing device that
* can directly control the objects on screen, and even draw or write on it. It is based on computer vision,
* machine learning and pattern recognition.
* The demo videos are here:
* http://www.youtube.com/watch?v=LUyql0SVobc
* http://www.youtube.com/watch?v=ExE5m6BjnV0
* 
* Author: Ben Wu
* Email: benwu232 at gmail dot com
* License: MIT license
*****************************************************************************/


#include <opencv2/contrib/contrib.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <iomanip>
#include <stdio.h>

//#define __linux 1

#if __linux
	#include <unistd.h>
	#include <X11/X.h>
	#include <X11/Xlib.h>
	#include <X11/Xutil.h>
#endif

#if WIN32
#include <windows.h>
#include <WinUser.h>

#endif

#define HAVE_TBB 1
#define HAVE_IPP 0
#define CV_HAAR_USE_SSE 1

#define USING_CUDA  0
#define DISPLAY_INFO 1

#if USING_CUDA
#include <opencv2/gpu/gpu.hpp>
using namespace cv::gpu;
#endif

int MinDetectWidth = 12, MinDetectHeight = 16;
int MaxDetectWidth = 48, MaxDetectHeight = 64;

using namespace std;
using namespace cv;

void help()
{
    cout << "Usage: ./qp -cam camid -cascade thecascade.xml\n"               
            "Using OpenCV version " << CV_VERSION << endl << endl;
}

string cascade_name = " ";

const float STEP = 0.001;
const int PERIOD = 100;
const int PERIOD_SMALL = 50;
const int STEP_BIG = 1;
const int STEP_SMALL = 0;
const int BIG_THRESHOLD = 4;
const int MAX_SIZE = 300;

struct dd
{
	int x;
	int y;
};


int err_pos_sample_width = 36;
int err_pos_sample_height = 48;
const char* scale_opt = "--scale=";
int scale_opt_len = (int)strlen(scale_opt);
const char* nested_cascade_opt = "--nested-cascade";
int nested_cascade_opt_len = (int)strlen(nested_cascade_opt);
int i;
const char* input_name = 0;
int WinSize = 300;

string inputName = "0";


#if __linux

Display *display;
Window root;

void initX()
{
	if ((display = XOpenDisplay(NULL)) == NULL) {
		fprintf(stderr, "Cannot open local X-display.\n");
		return;
	}
	root = DefaultRootWindow(display);
}

void GetCursorPos(int &x,int &y)
{
    int tmp = 0;
    unsigned int tmp2 = 0;
    Window fromroot, tmpwin;
    XQueryPointer(display, root, &fromroot, &tmpwin, &x, &y, &tmp, &tmp, &tmp2);
}

void SetCursorPos(int x,int y)
{
    XWarpPointer(display, None, root, 0, 0, 0, 0, x, y);
    XFlush(display);
}
#endif

#if WIN32
void GetCursorPos(int &x,int &y)
{
	POINT pt;

	GetCursorPos(&pt);
	x = pt.x;
	y = pt.y;
}
#endif


int sign (int n)
{
    if (n > 4)
        return 1;
    else if (n < -4)
        return -1;
    else
        return 0;
}
#if 0
void UpdateCurPos (int dx_cam, int dy_cam)
{
	int x = 0, y = 0;
    static float theta = 0.08;

    static int cnt = 0, cnt_small = 0;
    static int sum = 0, sum0 = 1000000000;
    static int sum_small = 0, sum0_small = 1000000000, ring_len = 8, ring_cur = 0;
    static int state = STEP_SMALL, old_state  = STEP_SMALL, state_cnt = 0;
    static dd ring[MAX_SIZE];
    int ring_idx = 0, k = 0;

    old_state = state;

    if (cnt == 1000000000) 
        cnt = 1000;
    
    int dx = int(dx_cam*theta);
	int dy = int(dy_cam*theta);

	if (dx == 0 && dy == 0)
	{
		state_cnt --;
		if (state_cnt < 0)
			state_cnt = 0;
		if (state_cnt == 0)
			state = STEP_SMALL;
	}
	else
	{
		state_cnt ++;
		if (state_cnt > BIG_THRESHOLD)
			state_cnt = BIG_THRESHOLD;
		if (state_cnt == BIG_THRESHOLD)
			state = STEP_BIG;
	}

    if (state == STEP_BIG)
    {
    	cnt++;
		sum += (dx_cam*dx_cam + dy_cam*dy_cam);

		if (cnt % PERIOD == PERIOD - 1)
		{
			if (sum > sum0)
				//direction = 0 - direction;
				theta -= STEP;
			else
				theta += STEP;
			printf("\n%8d%16d%16f\n", sum, sum0, theta);
			sum0 = sum;
			sum = 0;
		}
	}
    else // state == STEP_SMALL
    {
    	cnt_small ++;
    	sum_small += (dx_cam*dx_cam + dy_cam*dy_cam);
    	if (cnt_small % PERIOD_SMALL == PERIOD_SMALL - 1)
		{
			if (sum_small > sum0_small)
				ring_len++;
			else if (ring_len > 1)
				ring_len--;
			sum0_small = sum_small;
			sum_small = 0;
			printf("\nring_len = %8d\n", ring_len);
		}

    	// ring_cur++
    	if (ring_cur == MAX_SIZE-1)
    		ring_cur = 0;
    	else
    		ring_cur++;

    	ring[ring_cur].x = dx_cam;
    	ring[ring_cur].y = dy_cam;

    	dx = 0; dy = 0; ring_idx = ring_cur;
    	for (k = 0; k < ring_len; k++)
    	{
    		dx += ring[ring_idx].x;
    		dy += ring[ring_idx].y;

    		if (ring_idx == 0)
    			ring_idx = MAX_SIZE-1;
    		else
    			ring_idx--;
    	}

    	dx /= (ring_len*6);
    	dy /= (ring_len*6);
    }

    GetCursorPos(x, y);

    SetCursorPos(x+dx, y+dy);
    //printf ("%8d%8d\t%8d%8d\t%f\n", dx_cam, int(dx_cam*theta), dy_cam, int(dy_cam*theta), theta);
}
#endif

#if 1
void UpdateCurPos (int dx_cam, int dy_cam)
{
	int x = 0, y = 0;
    static float theta = 0.08;

    static int cnt = 0, cnt0 = 0;
    static int sum = 0, sum0 = 1000000000;

    cnt++;

    if (cnt == 1000000000)
        cnt = 500;

    sum += (dx_cam*dx_cam + dy_cam*dy_cam);

    if (cnt % PERIOD == PERIOD - 1)
    {
        if (sum > sum0)
            //direction = 0 - direction;
            theta -= STEP;
        else
            theta += STEP;
        //printf("\n%8d%16d%16f\n", sum, sum0, theta);
        sum0 = sum;
        sum = 0;
    }

    GetCursorPos(x, y);

    int dx = int(dx_cam*theta);
    int dy = int(dy_cam*theta);

	if (dx == 0 && dy == 0)
		cnt0 ++;
	else
		cnt0 = 0;

	if (cnt0 > 3)
	{
		dx = dx_cam / 6.0f;
		dy = dy_cam / 6.0f;
	}

/*

    if (dx == 0 && dx_cam > 4)
        dx = 1;

    if (dy == 0 && dy_cam > 4)
        dy = 1;
*/
    SetCursorPos(x+dx, y+dy);
    //printf ("%8d%8d\t%8d%8d\t%f\n", dx_cam, int(dx_cam*theta), dy_cam, int(dy_cam*theta), theta);
}
#endif

template<class T> void convertAndReseize(const T& src, T& gray, T& resized, double scale = 2.0)
{
    if (src.channels() == 3)
        cvtColor( src, gray, CV_BGR2GRAY );
    else
        gray = src;

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));
    if (scale != 1)
        resize(gray, resized, sz);
    else
        resized = gray;
}

int qp ()
{
    int dx, dy;
    int continuous = 0;
	int IsCursorShow = 1;
    double angle[] = {0, -10.0, 10.0};
    //float m[6];
    int k;
	int IsShow = 1;
            
#if 0
    ////////////////////////////////////////////////////////////////
    cv::gpu::DeviceInfo MyGpu;
    printf("Stream processor number: %d\n", MyGpu.multiProcessorCount());
    printf("Free GPU memory: %d\n", MyGpu.freeMemory());
    printf("Total GPU memory: %d\n", MyGpu.totalMemory());
    cvWaitKey(0);
    //return 1;
#endif

#if USING_CUDA
    if (cv::gpu::getCudaEnabledDeviceCount() == 0)
        return cerr << "No GPU found or the library is compiled without GPU support" << endl, -1;
#endif

    VideoCapture capture;
     
    //string cascade_name = argv[1];
    //string inputName = argv[2];

#if USING_CUDA     // if use LBP, this shoule be 0
    cv::gpu::CascadeClassifier_GPU cascade_gpu;
    if( !cascade_gpu.load( cascade_name ) )
        return cerr << "ERROR: Could not load cascade classifier \"" << cascade_name << "\"" << endl, help(), -1;
#endif

    cv::CascadeClassifier cascade_cpu;
    if( !cascade_cpu.load( cascade_name ) )
	{
		cout << "Can't load "<< cascade_name << endl;
        return cerr << "ERROR: Could not load cascade classifier \"" << cascade_name << "\"" << endl, help(), -1;
	}
	Mat image = imread( inputName);
	if( image.empty() )
    //printf("@@@@@@@\n"); fflush(stdout);
        if (!capture.open(inputName))
        {
    //printf("xxxxxx\n"); fflush(stdout);
            int camid = 0;
            sscanf(inputName.c_str(), "%d", &camid);
            if(!capture.open(camid))
                cout << "Can't open source" << endl;
        }

//#if DISPLAY_INFO 
#if 1
    namedWindow( "result", 1 );        
#endif

    Mat frame, frame_cpu, frame_copy, gray_cpu, resized_cpu, faces_downloaded, frameDisp, frame_roi, rot_img;
    vector<Rect> facesBuf_cpu;

#if USING_CUDA
    GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;
    bool useGPU = false;
#endif

    /* parameters */
    double scale_factor = 1;
    double font_scale = 0.8;
    
    bool visualizeInPlace = false;   
    bool findLargestObject = true;    
    int minNeighbors = 3;
#if 0 //USING_CUDA
    printf("\t<space> - toggle GPU/CPU\n");
    printf("\tV       - toggle visualisation in-place (for GPU only)\n");
    printf("\tL       - toggle lagest faces\n");
    printf("\t1/q     - inc/dec scale\n");
#endif
        
    int detections_num;
    for(;;)
    {               
        if( capture.isOpened() )
        {
            capture >> frame;                            
            if( frame.empty())
                break;
        }

        (image.empty() ? frame(Rect((frame.cols - WinSize)/2, (frame.rows - WinSize)/2, 
                          WinSize, WinSize)) : image).copyTo(frame_cpu);        // Set ROI
        //(image.empty() ? frame : image).copyTo(frame_cpu);
#if USING_CUDA
        frame_gpu.upload( image.empty() ? frame(Rect((frame.cols - WinSize)/2, (frame.rows - WinSize)/2, 
                          WinSize, WinSize)) : image);
        convertAndReseize(frame_gpu, gray_gpu, resized_gpu, scale_factor);
#endif
        convertAndReseize(frame_cpu, gray_cpu, resized_cpu, scale_factor);

        //frame_copy = resized_cpu;
        resized_cpu.copyTo (frame_copy);
        
        cv::TickMeter tm;
        tm.start();      
        
        for (k = 0; k < 3; k++)
        {
            /*
            // Matrix m looks like:
            //
            // [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
            // [ m3  m4  m5 ]       [ A21  A22   b2 ]
            //
            CvMat M = cvMat (2, 3, CV_32F, m);
            m[0] = (float) (cos (-angle[k] * 2 * CV_PI / 360.));
            m[1] = (float) (sin (-angle[k] * 2 * CV_PI / 360.));
            m[3] = -m[1];
            m[4] = m[0];

            // Compute rotation matrix
            CvPoint2D32f center = cvPoint2D32f( resized_cpu.cols/2, resized_cpu.rows/2 );
            cv2DRotationMatrix( center, angle, scale, rot_mat );

            cvGetQuadrangleSubPix (resized_cpu, rot_img, &M);
            //cvShowImage( "result", dst );
            //cvWaitKey(0);
*/
            //Matrix rotation
            Point2f src_center(resized_cpu.cols/2.0F, resized_cpu.rows/2.0F);
            Mat rot_mat = getRotationMatrix2D(src_center, angle[k], 1.0);
            warpAffine(resized_cpu, rot_img, rot_mat, resized_cpu.size());
                    
    #if USING_CUDA
            if (useGPU)
            {
                cascade_gpu.visualizeInPlace = visualizeInPlace;   
                cascade_gpu.findLargestObject = findLargestObject;    

                detections_num = cascade_gpu.detectMultiScale( resized_gpu, facesBuf_gpu, 1.1, minNeighbors, cvSize(MinDetectWidth, MinDetectHeight)); 
                facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
            }
            else /* so use CPU */
    #endif
            {   
                //Size minSize = cascade_gpu.getClassifierSize();
                //if (findLargestObject)
                {                
                    /*float ratio = (float)std::min(frame.cols / minSize.width, frame.rows / minSize.height);
                    ratio = std::max(ratio / 2.5f, 1.f);
                    minSize = Size(cvRound(minSize.width * ratio), cvRound(minSize.height * ratio));                
    */
                    //float ratio = (float)std::min(frame.cols / float(MinDetectWidth), frame.rows / float(MinDetectHeight));
                    //ratio = std::max(ratio / 2.5f, 1.f);
                    //minSize = Size(cvRound(MinDetectWidth * ratio), cvRound(MinDetectHeight * ratio));                

                }
#if 1           
				//cascade_cpu.detectMultiScale(rot_img, facesBuf_cpu, 1.2, minNeighbors, (findLargestObject ? CV_HAAR_FIND_BIGGEST_OBJECT : 0) | CV_HAAR_SCALE_IMAGE, cvSize(MinDetectWidth, MinDetectHeight));
                //cascade_cpu.detectMultiScale(rot_img, facesBuf_cpu, 1.2, minNeighbors, (findLargestObject ? CV_HAAR_FIND_BIGGEST_OBJECT : 0) | CV_HAAR_SCALE_IMAGE, cvSize(MinDetectWidth, MinDetectHeight), cvSize(MaxDetectWidth, MaxDetectHeight));
				cascade_cpu.detectMultiScale(rot_img, facesBuf_cpu, 1.2, minNeighbors, (findLargestObject ? CV_HAAR_FIND_BIGGEST_OBJECT : 0), cvSize(MinDetectWidth, MinDetectHeight), cvSize(MaxDetectWidth, MaxDetectHeight));
				//cascade_cpu.detectMultiScale(rot_img, facesBuf_cpu, 1.2, minNeighbors, CV_HAAR_DO_CANNY_PRUNING, cvSize(MinDetectWidth, MinDetectHeight), cvSize(MaxDetectWidth, MaxDetectHeight));
				//cascade_cpu.detectMultiScale(rot_img, facesBuf_cpu, 1.2, minNeighbors, CV_HAAR_DO_ROUGH_SEARCH, cvSize(MinDetectWidth, MinDetectHeight), cvSize(MaxDetectWidth, MaxDetectHeight));
				
#else
	            CvSeq* faces = cvHaarDetectObjects( rot_img, cascade, storage,
                                    1.2, 3, 0
                                    |CV_HAAR_FIND_BIGGEST_OBJECT
                                    //|CV_HAAR_DO_ROUGH_SEARCH
                                    //|CV_HAAR_DO_CANNY_PRUNING
                                    //|CV_HAAR_SCALE_IMAGE
                                    ,
                                    cvSize(12, 16) );

            
				for( int i = 0; i < (faces ? faces->total : 0); i++ )
				{
					CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
					//CvMat small_img_roi;
					//CvSeq* nested_objects;
	                
					CvScalar color = colors[i%8];
					//int radius;
					center.x = cvRound((r->x + r->width*0.5)*scale);
					center.y = cvRound((r->y + r->height*0.5)*scale);
					//radius = cvRound((r->width + r->height)*0.25*scale);
					//cvCircle( img, center, radius, color, 3, 8, 0 );
					//printf("%4d%4d\n", center, radius);

					face_rect = *(CvRect*)cvGetSeqElem( faces, i );
					cvRectangle( img, cvPoint(int(face_rect.x*scale),int(face_rect.y*scale)),
						cvPoint(int((face_rect.x+face_rect.width)*scale), int((face_rect.y+face_rect.height)*scale)),
						CV_RGB(255,0,0), 3 );

					goto NEXT;    
				 }

#endif
                detections_num = (int)facesBuf_cpu.size();
            }
                if (detections_num)
                    break;
        }

        tm.stop();
#if 0
//#if DISPLAY_INFO
        printf( "detection time = %g ms\t", tm.getTimeMilli() );
#endif

#if USING_CUDA
        if (useGPU)
            resized_gpu.download(resized_cpu);
        if (!visualizeInPlace || !useGPU)
#endif
            if (detections_num)
            {
                //tm.start();
#if USING_CUDA     // if use LBP, this shoule be 0
                Rect* faces = useGPU ? faces_downloaded.ptr<Rect>() : &facesBuf_cpu[0];                
#else
                Rect* faces = &facesBuf_cpu[0];
#endif
                for(int i = 0; i < detections_num; ++i)                
                    cv::rectangle(resized_cpu, faces[i], Scalar(255));            

                dx = WinSize/2 - (faces[0].x + faces[0].width/2);
                dy = WinSize/2 - (faces[0].y + faces[0].height/2);

#if 1
                tm.start();
                //UpdateCurPos2 (dx, dy);
                UpdateCurPos (dx, dy);

                tm.stop();
#else
				//UpdateCurPos (dx, dy);

				POINT HotSpot;
				//HCURSOR hCursor = LoadCursor(NULL , IDC_ARROW);
				HCURSOR hCursor = LoadCursor(NULL , IDC_NO);
                printf("cursor = %ld\n", (long)hCursor);
				GetCursorPos(&HotSpot);
				printf("HotSpot(%d, %d)\n", HotSpot.x, HotSpot.y);
				HWND WinHdl = WindowFromPoint(HotSpot);
				printf("WinHdl = %ld\n", (long)WinHdl);
				//SetClassLong(WinHdl, GCL_HCURSOR, (LONG)LoadCursor(NULL , IDC_CROSS));
				SetClassLong(WinHdl, GCL_HCURSOR, (LONG)hCursor);
				//SetClassLong(WinHdl, GCL_HCURSOR, hCursor);
				HCURSOR PreCur = SetCursor(hCursor);
#endif

#if 0
//#if DISPLAY_INFO
                printf( "Cursor time = %g ms\t", tm.getTimeMilli() );
#endif
            }
#if 0
			else
			{
				POINT HotSpot;
				HCURSOR hCursor = LoadCursor(NULL , IDC_CROSS);
                //printf("cursor = %ld\n", (long)hCursor);
				GetCursorPos(&HotSpot);
				//printf("HotSpot(%d, %d)\n", HotSpot.x, HotSpot.y);
				HWND WinHdl = WindowFromPoint(HotSpot);
				//printf("WinHdl = %ld\n", (long)WinHdl);
				SetClassLong(WinHdl, GCL_HCURSOR, (LONG)LoadCursor(NULL , IDC_CROSS));
				//SetClassLong(WinHdl, GCL_HCURSOR, (LONG)hCursor);
				//SetClassLong(WinHdl, GCL_HCURSOR, hCursor);
				HCURSOR PreCur = SetCursor(hCursor);

			}
#endif

#if 0
//#if DISPLAY_INFO
            printf ("\n");
#endif
        
        int tickness = font_scale > 0.75 ? 2 : 1;

        Point text_pos(5, 25);        
        Scalar color = CV_RGB(255, 0, 0);
        Size fontSz = cv::getTextSize("T[]", FONT_HERSHEY_SIMPLEX, font_scale, tickness, 0);
        int offs = fontSz.height + 5;

        cv::cvtColor(resized_cpu, frameDisp, CV_GRAY2BGR);
#if DISPLAY_INFO
        char buf[4096];
        //sprintf(buf, "%s, FPS = %0.3g", useGPU ? "GPU (device) " : "CPU (host)", 1.0/tm.getTimeSec());                       
        //sprintf(buf, "%d FPS = %0.3g", WinSize, 1.0/tm.getTimeSec());                       
        sprintf(buf, "%d %0.3g", WinSize, 1.0/tm.getTimeSec());                       
        putText(frameDisp, buf, text_pos, FONT_HERSHEY_SIMPLEX, font_scale, color, tickness);
        //sprintf(buf, "scale = %0.3g,  [%d x %d] x scale, Min neighbors = %d", scale_factor, frame.cols, frame.rows, minNeighbors);                       
        //putText(frameDisp, buf, text_pos+=Point(0,offs), FONT_HERSHEY_SIMPLEX, font_scale, color, tickness);
        //putText(frameDisp, "Hotkeys: space, 1/Q, 2/E, 3/E, L, V, Esc", text_pos+=Point(0,offs), FONT_HERSHEY_SIMPLEX, font_scale, color, tickness);
#endif
        //if (findLargestObject)
          //  putText(frameDisp, "FindLargestObject", text_pos+=Point(0,offs), FONT_HERSHEY_SIMPLEX, font_scale, color, tickness);
#if USING_CUDA
        if (visualizeInPlace && useGPU)
            putText(frameDisp, "VisualizeInPlace", text_pos+Point(0,offs), FONT_HERSHEY_SIMPLEX, font_scale, color, tickness);
#endif
        if (continuous)
            putText(frameDisp, "Continuously negative", text_pos+Point(0,offs), FONT_HERSHEY_SIMPLEX, font_scale, color, tickness);

        int key = waitKey( 1 );
        if( key == 27)
            break;

        switch ((char)key)
        {
#if USING_CUDA
			case ' ':  useGPU = !useGPU;  printf("Using %s\n", useGPU ? "GPU" : "CPU");break;
#endif
			case 'v':  case 'V': visualizeInPlace = !visualizeInPlace; printf("VisualizeInPlace = %d\n", visualizeInPlace); break;
			case 'l':  case 'L': findLargestObject = !findLargestObject;  printf("FindLargestObject = %d\n", findLargestObject); break;
			case '9':  scale_factor*=1.05; printf("Scale factor = %g\n", scale_factor); break;
			case 'o':  case 'Q':scale_factor/=1.05; printf("Scale factor = %g\n", scale_factor); break;

			case '3':  font_scale*=1.05; printf("Fond scale = %g\n", font_scale); break;
			case 'e':  case 'E':font_scale/=1.05; printf("Fond scale = %g\n", font_scale); break;

			case '2':  ++minNeighbors; printf("Min Neighbors = %d\n", minNeighbors); break;
			case 'w':  case 'W':minNeighbors = max(minNeighbors-1, 0); printf("Min Neighbors = %d\n", minNeighbors); break;

			case '4':  WinSize += 10; break;
			case 'r':  WinSize -= 10; break;

        }
#if DISPLAY_INFO
        rectangle(frameDisp, 
                    cvPoint((frameDisp.cols - err_pos_sample_width)/2, (frameDisp.rows - err_pos_sample_height)/2), 
                    cvPoint((frameDisp.cols + err_pos_sample_width)/2-1, (frameDisp.rows + err_pos_sample_height)/2-1), 
                    cvScalar(0,255,0));
        cv::imshow( "result", frameDisp);
#endif

       
    }    

    return 0;
}

//int main( int argc, const char** argv )
int main(int argc, char *argv[])
{        
#if __linux
	initX();
#endif

    for( int i = 1; i < argc; i++ )
    {
        if( !strcmp( argv[i], "-err-pos-sample-width" ) )
        {
            err_pos_sample_width = atoi( argv[++i] );
			MaxDetectWidth = err_pos_sample_width;
        }
        else if( !strcmp( argv[i], "-err-pos-sample-height" ) )
        {
            err_pos_sample_height = atoi( argv[++i] );
			MaxDetectHeight = err_pos_sample_height;
        }
        else if (!strcmp( argv[i], "-win-size" ))
        {
            WinSize = atoi(argv[++i]);
        }
		else if (!strcmp( argv[i], "-cascade" ))
		{
			cascade_name = argv[++i];
		}
		else if (!strcmp( argv[i], "-cam" ))
		{
			inputName = argv[++i];
		}

			else if( argv[i][0] == '-' )
        {
            fprintf( stderr, "WARNING: Unknown option %s\n", argv[i] );
        }
        else
            input_name = argv[i];
	}

	printf ("\n\n*************************  Quantum Phantom  *************************\n\n");
    printf ("                       -- proudly present by Ben Wu  \n\n");

    qp();
    return 0;
}



