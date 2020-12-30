#include <vector>
#include <opencv4/opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <glob.h>
#include <iostream>
#include "class_timer.hpp"
#include "class_detector.h"
#include <opencv2/highgui/highgui_c.h>
#include <sstream>
#include <map>
#include <string>
#include <fstream>
#include <cstdio>
#include <iomanip>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "dirent.h"
#include "IOUT.h"
//#include "UA-DETRAC.h"


#include <memory>
#include <thread>
#include "IOUT.h"
#define USE_IN_DETRAC 0
#define SHOW_BOXES 1 // Program will output boxes after finish tracking
#define SAVE_BOXES 1 // Save the boxes on image file


using namespace cv;
using namespace std;

int main()
{
    Config config_v4;
    config_v4.net_type = YOLOV4;
    config_v4.file_model_cfg = "../configs/yolov4.cfg";
    config_v4.file_model_weights = "../configs/yolov4.weights";
    config_v4.inference_precison = FP32;

    Config config_v4_tiny;
    config_v4_tiny.net_type = YOLOV4_TINY;
    config_v4_tiny.detect_thresh = 0.5;
    config_v4_tiny.file_model_cfg = "../data/yolov4-tiny-obj.cfg";
    config_v4_tiny.file_model_weights = "../data/yolov4-tiny.weights";
    config_v4_tiny.calibration_image_list_file_txt = "../configs/images.txt";
    config_v4_tiny.inference_precison = FP32;

    std::unique_ptr<Detector> detector(new Detector());
    detector->init(config_v4_tiny);
    map<int, string> mymap;


    mymap[0] = "car";
    mymap[1] = "person_sb";
    mymap[2] = "person_nsb";
    mymap[3] = "helmet";
    mymap[4] = "no_helmet";
    mymap[5] = "phone";
    mymap[6] = "no_phone";
    mymap[7] = "tr";
    mymap[8] = "notr";
    mymap[9] = "truck";
    mymap[10] = "bus";
    mymap[11] = "auto";
    mymap[12] = "ped";
    mymap[13] = "bicycle";
    mymap[14] = "phone_right";
    mymap[15] = "phone_left";



    vector<String> fn;
    std::vector< Track > drawing_tracks;
    std::vector< Track > tracks;
    std::vector<BoundingBox>det;
    std::vector<std::vector<BoundingBox> > detections;
    float sigma_l = 0;		// low detection threshold
    float sigma_h = 0.2;		// high detection threshold
    float sigma_iou = 0.5;	// IOU threshold
    float t_min = 2;
    string videoPath = "/home/srikar/Documents/traffic.mp4";
    cv::VideoCapture cap(videoPath);
    cv::Mat frame1;
    if(!cap.isOpened())
    {
        cout << "Error opening video file " << videoPath << endl;
        return -1;
    }
    while(cap.isOpened())
    {
        cap >> frame1;
        if (frame1.empty()) break;
        Timer timer;
        timer.reset();
        //cv::Mat image0 = cv::imread(frame, cv::IMREAD_UNCHANGED);
        std::vector<BatchResult> batch_res;
        //Timer timer;
        //prepare batch data
        std::vector<cv::Mat> batch_img;
        cv::Mat temp0 = frame1.clone();
        batch_img.push_back(temp0);

        //detect
        //timer.reset();
        detector->detect(batch_img, batch_res);
        timer.out("detect");
        //disp
        for (int i=0;i<batch_img.size();++i)
        {
            for (const auto &r : batch_res[i])
            {
                struct BoundingBox b;
                b.x = r.x;
                b.y = r.y;
                b.w = r.w;
                b.h = r.h;
                b.score = r.prob;
                det.push_back(b);
            }
            detections.push_back(det);
                
            for (int i=0; i<detections.size();i++)
            {
                for (int j=0;j<det.size();j++)
                {
                    cout << det[j].x << "\n"
                         << det[j].y << "\n"
                         << det[j].w << "\n"
                         << det[j].h << "\n"
                         << det[j].score << "\n";                            
                }
            }

            tracks = track_iou(sigma_l, sigma_h, sigma_iou, t_min, detections);
            std::cout << "Last Track ID > " << tracks.back().id << std::endl;
                
            #if USE_IN_DETRAC
            #if SHOW_BOXES
            std::cout << "Displaying results on window..." << std::endl;
            /// Show results on image
            cv::Mat image;
            cv::namedWindow("Display Tracking", cv::WINDOW_AUTOSIZE);
            for (int frame = 0; frame < detections.size(); frame++)
            {
                // Load the current image
                //image = cv::imread(frame1);
                image = frame1;
                // Grab all the tracks that start in current frame
                for (auto track : tracks)
                {
                    if (track.start_frame == frame)
                    {
                        drawing_tracks.push_back(track);
                    }
                }
                    // Write all the boxes into the image
                for (auto dt : drawing_tracks)
                {
                    int box_index = frame - dt.start_frame;
                    if (box_index < dt.boxes.size() )
                    {
                        BoundingBox b = dt.boxes[box_index];
                        cv::rectangle(image, cv::Point(b.x, b.y), cv::Point(b.x + b.w, b.y + b.h), cv::Scalar(0, 0, 255), 2);
                        cv::putText(image, std::to_string(dt.id), cv::Point(b.x + b.w - b.w / 2, b.y + b.h - 5), 1, 1, cv::Scalar(0, 255, 255), 2);
                    }
                }
                #if SAVE_BOXES
                /// Save the images
                std::ostringstream name;
                //name << "/home/srikar/yolov4-tiny-trt/data/detections/image" << j << ".jpg";
                cv::imwrite(name.str(), image);
                #endif
                imshow("Display Tracking", image);
                cv::waitKey(50);
            }
            std::cout << "Displaying images finished!!" << std::endl;
            #endif //SHOW BOXES
            #else // detections.size() would be the amount of frames here
            #endif
        }
    }
}



	