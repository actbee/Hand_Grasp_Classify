# Hand Grasp Classify

## ABOUT THE TASK

We are asked to find the hand image from the screenshots of a First-Person Video (EGTEA Gaze+ dataset to be more specific) and then 
classify the grasp type of different hands seperately.

![avatar](https://github.com/actbee/Hand_Grasp_Classify/blob/master/img/task.png?raw=true)

## RUNNING ENVIRONMENT

We have run our UMA/ and classifyer/ part on Google Colab, which means they usually need a typical GPU environment, some mainstream deeplearning freameworks(like 
Pytorch here) and normal Python models (like pandas) together with Acaconda in Linux system like Ubuntu.

As for the Another_result, we developed our project thorugh C++ with Xcode on OSX. You shall need OpenCV2 environment to run A_grabcut/ and B_handtracker/.
OpenCV3 is needed to run C_classify. 

## WHOLE DOCUMENTS

You can download the whole documents(including some other images produced in this task) through
this link: https://pan.baidu.com/s/1zGeSSzMKpOd6dWMGApxMTw (password: cvpr)

## Statement

We reference some other works to our project， this may include:
 "Generalizing Hand Segmentation in Egocentric Videos with Uncertainty-Guided Model Adaptation" by Minjie Cai, Feng Lu and Yoichi Sato
 (https://github.com/actbee/UMA)
 
 "Pixel-level Hand Detection for Ego-centric Videos" by Cheng Li and Kris M. Kitani
 （https://github.com/irllabs/handtrack and https://github.com/cmuartfab/grabcut)
 
 "Hand Keypoint Detection in Single Images using Multiview Bootstrapping” by Tomas Simon, Hanbyul Joo, Iain Matthews, Yaser Sheikh
 (https://github.com/spmallick/learnopencv/tree/master/HandPose)
 
 We really appreciate them for their great works!