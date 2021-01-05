/**
 we once used this part of code as a way to recognise the hand from the original image,
 we labeld some original images using  grab-cut method provided.
 we trained the model with our labeld images and use the model to get the final resut.
 我们尝试了这种方法来分割手部图像。
 我们用了grabcut那个工具来打标我们的原始数据。
 我们用打标后的数据训练模型，然后喂入原图得到了最终分割的结果。
 */
#include <iostream>
#include <iomanip>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "FeatureComputer.hpp"
#include "Classifier.h"
#include "LcBasic.h"
#include "HandDetector.hpp"

using namespace std;
using namespace cv;

int main (int argc, char * const argv[])
{
    bool TRAIN_MODEL = 0;           //1 if you are training the models, 0 if you are running the program to predict
    bool TEST_MODEL  = 1;           //0 if you are training the models, 1 if you are running the program to predict
    
    int target_width = 320;			// for resizing the input (small is faster)
    
    // maximum number of image masks that you will use
    // must have the masks prepared in advance
    // only used at training time
    int num_models_to_train = 149;
    
    
    // number of models used to compute a single pixel response
    // must be less than the number of training models
    // only used at test time
    int num_models_to_average = 149;
    
    // runs detector on every 'step_size' pixels
    // only used at test time
    // bigger means faster but you lose resolution
    // you need post-processing to get contours
    int step_size = 3;
    
    // Assumes a certain file structure e.g., /root/img/basename/00000000.jpg
    string root = "/Users/actbee/Documents/Vision/final/handtrack-master/handtrack-master/";       //replace with path to your Xcode project
    string basename = "";
    string img_prefix		= root + "img"		+ basename + "/";			// color images
    string msk_prefix		= root + "mask"     + basename + "/";			// binary masks
    string model_prefix		= root + "models"	+ basename + "/";			// output path for learned models
    string globfeat_prefix  = root + "globfeat" + basename + "/";			// output path for color histograms
    
    
    // types of features to use (you will over-fit if you do not have enough data)
    // r: RGB (5x5 patch)
    // v: HSV
    // l: LAB
    // b: BRIEF descriptor
    // o: ORB descriptor
    // s: SIFT descriptor
    // u: SURF descriptor
    // h: HOG descriptor
    string feature_set = "rvl";
    
    
    
    if(TRAIN_MODEL)
    {
        cout << "Training...\n";
        HandDetector hd;
        hd.loadMaskFilenames(msk_prefix);
        hd.trainModels(basename, img_prefix, msk_prefix,model_prefix,globfeat_prefix,feature_set,num_models_to_train,target_width);
        cout << "Done Training...\n";
    }
    
    
    
    if(TEST_MODEL)
    {
        cout << "Testing...\n";
        string vid_filename		= root + "vid/"		+ basename + ".avi";
        string task=root+"task/images/";
        string result=root+"task/result/";
        
       // VideoCapture cap(0);
        Mat im;
        Mat ppr;
        
       // VideoWriter avi;
        stringstream ss;
        ss.str("");
        ss << "mkdir -p " + result;
        system(ss.str().c_str());
        vector<int> q(2,100);
        q[0]=1;
        
        
        /*
        Mat img;
        img=imread(task+"00000101.jpg");
        target_width=img.cols;
        HandDetector hd;
        hd.testInitialize(model_prefix,globfeat_prefix,feature_set,num_models_to_average,target_width);
        hd.test(img,num_models_to_average,step_size);
        Mat pp_contour = hd.postprocess(hd._response_img);        // binary contour
        Mat res;
        img.copyTo(res,pp_contour);
        */
        
       // hd.colormap(pp_contour,pp_contour,0);                    // colormap of contour
       // imshow("test",res);
      //  waitKey(0);
        Mat img;
        img=imread(task+"00000101.jpg");
        target_width=img.cols;
        HandDetector hd;
        hd.testInitialize(model_prefix,globfeat_prefix,feature_set,num_models_to_average,target_width);
        hd.test(img,num_models_to_average,step_size);
        
        int f=0;
        while(f<1600){
            f+=1;
            ss.str("");
            if(f>999){
            ss << task << setw(9) << setfill('0') << f << ".jpg";  //change here to set the range (8 0-999, 9 1000+)
            }
            else{
                ss << task << setw(8) << setfill('0') << f << ".jpg";
            }
            cout <<"Opening: " << ss.str() << endl;
            im = imread(ss.str());
            if(!im.data) {cout <<"no image data " << ss.str() << endl; continue;} // break
            
            hd.test(im,num_models_to_average,step_size);
            Mat pp_contour = hd.postprocess(hd._response_img);        // binary contour
         //   hd.colormap(pp_contour,pp_contour,0);                    // colormap of contour
            Mat res;
            im.copyTo(res,pp_contour);

            
            ss.str("");
            ss << result << setw(8) << setfill('0') << f << ".jpg";
            imwrite(ss.str(),res,q);
        }
        
        
        /*while(1)
        {
            cap >>(im);
      
            if(!im.data) break;
            //cap >> im; if(!im.data) break; // skip frames with these
            //cap >> im; if(!im.data) break;
            //cap >> im; if(!im.data) break;
            
            hd.test(im,num_models_to_average,step_size);
            
            
            // Different ways to visualize the results
            // hd._response_img (float probabilities in a matrix)
            // hd._blur (blurred version of _response_img)
            
            
            int SHOW_RAW_PROBABILITY = 0;
            if(SHOW_RAW_PROBABILITY)
            {
                Mat raw_prob;
                hd.colormap(hd._response_img,raw_prob,0);
                imshow("probability",raw_prob);	// color map of probability
            }
            
            int SHOW_BLUR_PROBABILITY = 0;
            if(SHOW_BLUR_PROBABILITY)
            {
                Mat pp_res;
                hd.postprocess(hd._response_img);
                imshow("blurred",hd._blu);		// colormap of blurred probability
            }
            
            int SHOW_BINARY_CONTOUR = 1;
            if(SHOW_BINARY_CONTOUR)
            {
                Mat pp_contour = hd.postprocess(hd._response_img);		// binary contour
                hd.colormap(pp_contour,pp_contour,0);					// colormap of contour
                imshow("contour",pp_contour);
            }
            
            int SHOW_RES_ALPHA_BLEND = 1;
            if(SHOW_RES_ALPHA_BLEND)
            {
                Mat pp_res = hd.postprocess(hd._response_img);
                hd.colormap(pp_res,pp_res,0);
                resize(pp_res,pp_res,im.size(),0,0,INTER_LINEAR);
                addWeighted(im,0.7,pp_res,0.3,0,pp_res);				// alpha blend of image and binary contour
                imshow("alpha_res",pp_res);
                
            }
            
            
            /*			
             if(!avi.isOpened())
             {
             stringstream ss;
             ss.str("");
             ss << root + "/vis/" + basename + "_skin.avi"; 
             int fourcc = avi.fourcc('F','L','V','1');
             avi.open(ss.str(),fourcc,30,ppr.size(),true);
             }
             avi << ppr;
             
            
            
            waitKey(1);
        }
         */
    }
    return 0;
}

