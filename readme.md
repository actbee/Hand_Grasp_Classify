# Hand Grasp Classify

![avatar](https://github.com/actbee/Hand_Grasp_Classify/blob/master/img/task.png?raw=true)

## ABOUT THE TASK

We are asked to find the hand image from the screenshots of a First-Person Video (EGTEA Gaze+ dataset to be more specific) and then 
classify the grasp type of different hands seperately.  

Our method composes two parts: A.Find out the hand from the images and B. Classify hands by their grasp gestures.  

![img1](https://github.com/actbee/actbee.github.io/blob/master/images/hand_grasp.png?raw=true)

We use two methods for each part in order to finish this task. Both covers a traditional method and a deep-learning based method.
The result shows some kind of superiority of deep-learning based method over the traditional method.
You can find more details through our [report](https://github.com/actbee/Hand_Grasp_Classify/blob/master/report.pdf) and our [slides](https://github.com/actbee/Hand_Grasp_Classify/blob/master/final.pdf) (currently both Chinese only).

![img2](https://github.com/actbee/actbee.github.io/blob/master/images/hand_grasp_2.png?raw=true)

This project has great potential future use into field like Human-Computer Interaction and also can be used to create some creative works.

## RUNNING ENVIRONMENT

We have run our UMA/ and classifyer/ part on Google Colab, which means they usually need a typical GPU environment, some mainstream deeplearning freameworks(like 
Pytorch here) and normal Python models (like pandas) together with Acaconda in Linux system like Ubuntu.

As for the Another_result, we developed our project thorugh C++ with Xcode on OSX. You shall need OpenCV2 environment to run A_grabcut/ and B_handtracker/.
OpenCV3 is needed to run C_classify/. 

## WHOLE DOCUMENTS

You can download the whole documents(including some other images produced in this task) through
this link: [https://pan.baidu.com/s/1zGeSSzMKpOd6dWMGApxMTw](https://pan.baidu.com/s/1zGeSSzMKpOd6dWMGApxMTw) (password: cvpr)

## STATEMENT

We reference some other works to our project，this may include:  
 "Generalizing Hand Segmentation in Egocentric Videos with Uncertainty-Guided Model Adaptation" by Minjie Cai, Feng Lu and Yoichi Sato  
 [https://github.com/actbee/UMA](https://github.com/actbee/UMA)
 
 "Pixel-level Hand Detection for Ego-centric Videos" by Cheng Li and Kris M. Kitani    
 ([https://github.com/irllabs/handtrack](https://github.com/irllabs/handtrack) and [https://github.com/cmuartfab/grabcut](https://github.com/cmuartfab/grabcut))
 
 "Hand Keypoint Detection in Single Images using Multiview Bootstrapping” by Tomas Simon, Hanbyul Joo, Iain Matthews, Yaser Sheikh  
  [https://github.com/spmallick/learnopencv/tree/master/HandPose](https://github.com/spmallick/learnopencv/tree/master/HandPose)
 
 We really appreciate them for their great works!