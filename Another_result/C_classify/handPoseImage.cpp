/*
 For the original input of this part of code is the images of hands cut from the original
 images but still not be cut into small rect regions.
Please notice that this part of code is mainly used for:
 1. cut the hand part from the original image into small rect region
 2. find the key points from the rect images
 3. calculate the center points of each hand based on the predicted key points
 4. cluster those center points based on k-means algorithm thus get the final result
 
 in fact, we change this file for multiple times to get the final classified result,
 which means that you may change some part of this code to run it successfully (but we
 are sure you can understand all of our method from this code. It has include all of the
 key algorithms. However, since running the whole code is quite time-confusing, we have
 not tested the whole code(we tested its different parts and successfully get the final
 results.
 If you meet any trouble while running this code, feel free to contact us.
 Really sorry for this confusing!
 
 这部分是我们用于将分割成单只手（大致）的图片得到最后分类结果的代码。
 这个代码的功能完整地包含了以下几个部分：
 1. 讲手部通过小的矩形区域从原图中分割出来，如此有利于后续关键点检测的正确率
 2. 寻找小矩形区域手部图像的关键点
 3. 计算每个手部关键点的中心
 4. 通过k-means算法对这些关键点聚类，得到最终的分类结果
 我们在这部分代码不断地删改得到了最终的分类结果，不过由于运行时间会很长，我们没有一次性跑过这个代码。但是这个
 代码上有我们过程写过的全部记录。如果你发现运行中出错，可以适当修改代码中的某些部分，或者直接联系我们。
 目前我们估计的潜在问题主要包括每一步骤生成的文件的命名方式上，因为我们中途手动修改过其中一些图片的名字，
 所以一次性测试此代码可能导致步骤间出现命名冲突而爆错的情况。
 我们对此造成的潜在困扰深感抱歉！

 */

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<iomanip>
#include<fstream>
#include<sstream>
#include <iostream>


using namespace std;
using namespace cv;
using namespace cv::dnn;


const int POSE_PAIRS[20][2] =
{
    {0,1}, {1,2}, {2,3}, {3,4},         // thumb
    {0,5}, {5,6}, {6,7}, {7,8},         // index
    {0,9}, {9,10}, {10,11}, {11,12},    // middle
    {0,13}, {13,14}, {14,15}, {15,16},  // ring
    {0,17}, {17,18}, {18,19}, {19,20}   // small
};

string protoFile = "/Users/actbee/Documents/Vision/final/CLASSIFY/classify/pose_deploy.prototxt";
string weightsFile = "/Users/actbee/Documents/Vision/final/CLASSIFY/classify/pose_iter_102000.caffemodel";
string result="/Users/actbee/Documents/Vision/final/CLASSIFY/images/points/";
string imageFile = "/Users/actbee/Documents/Vision/final/CLASSIFY/images/cut_img/";
string rectFile="/Users/actbee/Documents/Vision/final/CLASSIFY/images/rect_img/";
string txtFile="/Users/actbee/Documents/Vision/final/CLASSIFY/images/txt/";
string centerFile="/Users/actbee/Documents/Vision/final/CLASSIFY/images/";
string finalFile="/Users/actbee/Documents/Vision/final/CLASSIFY/images/final/";
int nPoints = 22;

//The following part is used for K-means
typedef struct Points{
    float x;
    float y;
    int cluster;
    Points (){}
    Points (float a,float b,int c){
        x = a;
        y = b;
        cluster = c;
    }
}points;
float stringToFloat(string i){
    stringstream sf;
    float score=0;
    sf<<i;
    sf>>score;
    return score;
}
vector<points> openFile(const char* dataset){
    fstream file;
    file.open(dataset,ios::in);
    vector<points> data;
    while(!file.eof()){
        string temp;
        file>>temp;
        int split = temp.find(',',0);
        points p(stringToFloat(temp.substr(0,split)),stringToFloat(temp.substr(split+1,temp.length()-1)),0);
        data.push_back(p);
    }
    file.close();
    return data;
}
float squareDistance(points a,points b){
    return (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y);
}
void k_means(vector<points> dataset,int k){
    vector<points> centroid;
    int n=1;
    int len = dataset.size();
    cout<<"the total data of kmeans is "<<len<<endl;
    srand((int)time(0));
    //random select centroids
    while(n<=k){
        int cen = (float)rand()/(RAND_MAX+1)*len;
        points cp(dataset[cen].x,dataset[cen].y,n);
        centroid.push_back(cp);
        n++;
    }
    for(int i=0;i<k;i++){
        cout<<"x:"<<centroid[i].x<<"\ty:"<<centroid[i].y<<"\tc:"<<centroid[i].cluster<<endl;
    }
    //cluster
    int time = 100;
    int oSSE = INT_MAX;
    int nSSE = 0;
    while(abs(oSSE-nSSE)>=1){
//    while(time){
        oSSE = nSSE;
        nSSE = 0;
        //update cluster for all the points
        for(int i=0;i<len;i++){
            n=1;
            float shortest = INT_MAX;
            int cur = dataset[i].cluster;
            while(n<=k){
                float temp=squareDistance(dataset[i],centroid[n-1]);
                if(temp<shortest){
                    shortest = temp;
                    cur = n;
                }
                n++;
            }
            dataset[i].cluster = cur;
        }
        //update cluster centroids
        int *cs = new int[k];
        for(int i=0;i<k;i++) cs[i] = 0;
        for(int i=0;i<k;i++){
            centroid[i] = points(0,0,i+1);
        }
        for(int i=0;i<len;i++){
            centroid[dataset[i].cluster-1].x += dataset[i].x;
            centroid[dataset[i].cluster-1].y += dataset[i].y;
            cs[dataset[i].cluster-1]++;
        }
        for(int i=0;i<k;i++){
            centroid[i].x /= cs[i];
            centroid[i].y /= cs[i];
        }
        cout<<"time:"<<time<<endl;
        for(int i=0;i<k;i++){
            cout<<"x:"<<centroid[i].x<<"\ty:"<<centroid[i].y<<"\tc:"<<centroid[i].cluster<<endl;
        }
        //SSE
        for(int i=0;i<len;i++){
            nSSE += squareDistance(centroid[dataset[i].cluster-1],dataset[i]);
        }
//        time--;
    }
    fstream clustering;
    clustering.open("/Users/actbee/Documents/Vision/final/CLASSIFY/images/clustering.txt",ios::out);
    for(int i=0;i<len;i++){
        clustering<<dataset[i].x<<","<<dataset[i].y<<","<<dataset[i].cluster<<"\n";
    }
    clustering.close();
//    cout<<endl;
//    for(int i=0;i<centroid.size();i++){
//        cout<<"x:"<<centroid[i].x<<"\ty:"<<centroid[i].y<<"\tc:"<<centroid[i].cluster<<endl;
//    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following part is used for calculte the center point of each rect hand image based on its key points
vector<string>split(string str,string pattern){
    vector<string> ret;
    if (pattern.empty()) return ret;
    size_t start = 0, index = str.find_first_of(pattern, 0);
    while (index != str.npos)
        {
            if (start != index)
                ret.push_back(str.substr(start, index - start));
            start = index + 1;
            index = str.find_first_of(pattern, start);
        }
    if (!str.substr(start).empty())
            ret.push_back(str.substr(start));
    return ret;
}

void get_center(){
    ifstream file;
    ofstream outfile;
    outfile.open(centerFile+"centerpoints.txt");
    int f=100;
    while(f<4000){
        f+=1;
        string num=to_string(static_cast<long long>(f));
        file.open(txtFile+"00000"+num+".txt");
        if(file.is_open()){
            cout<<"open "<<num<<" success"<<endl;
        }
        else{
            cout<<"no "<<num<<" !"<<endl;
            continue;
        }
        int sumx=0;
        int sumy=0;
        for(int i=0;i<22;i++){
            string content;
            string content2;
            if(i==0){
                file>>content;
                continue;
            }
            file>>content;
            vector<string>newcomer=split(content,",");
            int xadd=atoi(newcomer[0].c_str());
            sumx+=xadd;
            int yadd=atoi(newcomer[1].c_str());
            sumy+=yadd;
        }
        file.close();
        outfile<<sumx<<","<<sumy<<endl;
    }
    outfile.close();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//the following part is used for cut the original hand image into the rect image
Mat findhand(Mat& input){
    int col=input.cols;
    int row=input.rows;

    int uppest=9999;
    int downest=0;
    int leftest=9999;
    int rightest=0;
    
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            if(input.at<Vec3b>(i,j)[0]!=0||input.at<Vec3b>(i,j)[1]!=0||input.at<Vec3b>(i,j)[2]!=0){
                
                if(i>=downest){
                    downest=i;
                }
                if(i<=uppest){
                    uppest=i;
                }
                if(j>=rightest){
                    rightest=j;
                }
                if(j<=leftest){
                    leftest=j;
                }
            }
        }
    }
    Mat out;
    
    if(leftest==9999||uppest==9999){
        cout<<"blank"<<endl;
        input.copyTo(out);
        return out;
    }
    cout<<"begin rect"<<endl;
    cout<<"top left"<<leftest<<","<<uppest<<endl;
    cout<<"down right"<<rightest<<","<<downest<<endl;
    Rect rect(leftest,uppest,rightest-leftest,downest-uppest);
    out=input(rect);
    return out;
}


int main(int argc, char **argv)
{
    vector<int> q(2,100);
    q[0]=1;
    stringstream ss;
    
    ss.str("");
    ss << "mkdir -p " +imageFile;
    system(ss.str().c_str());
    
    
    cout << "USAGE : ./handPoseImage <imageFile> " << endl;
    // Take arguments from commmand line
    if (argc == 2)
    {
      imageFile = argv[1];
    }

    float thresh = 0.01;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // part1: cut the original hand image into the rect region
    int f=100;   //now read all of the original images
    while(f<4000)
    {
        
        f+=1;
        ss.str("");
        
       if(f>999){
            ss << imageFile << setw(9) << setfill('0') << f << ".png";  //change here to set the range (8 0-999, 9 1000+)
        }
        else{
           ss << imageFile << setw(8) << setfill('0') << f << ".png";
        }
        
        cout <<"Opening: " << ss.str() << endl;
        Mat frame = imread(ss.str());
        if(!frame.data) {cout <<"no image data " << ss.str() << endl; continue;} // break
        
    Mat cut=findhand(frame);   //to get the rect images based on the position of the hand
        
        ss.str("");
        ss << rectFile << setw(8) << setfill('0') << f << ".png";

        imwrite(ss.str(),cut,q);
    
    }
    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // part2: get the key points of hand from each rect image
    int f2=100;   //now read all of the rect images
    while(f2<4000)
    {
        f2+=1;
        ss.str("");
       if(f2>999){
            ss << rectFile << setw(9) << setfill('0') << f2 << ".png";
        }
        else{
            ss << rectFile << setw(8) << setfill('0') << f2 << ".png";
        }
        
        cout <<"Opening: " << ss.str() << endl;
        Mat frame = imread(ss.str());
        if(!frame.data) {cout <<"no image data " << ss.str() << endl; continue;} // break
        
    // below is the key point detection algortihm part
    Mat frameCopy = frame.clone();
    int frameWidth = frame.cols;
    int frameHeight = frame.rows;

    float aspect_ratio = frameWidth/(float)frameHeight;
    int inHeight = 368;
    int inWidth = (int(aspect_ratio*inHeight) * 8) / 8;

    cout << "inWidth = " << inWidth << " ; inHeight = " << inHeight << endl;

    double t = (double) cv::getTickCount();
    Net net = readNetFromCaffe(protoFile, weightsFile);

    Mat inpBlob = blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);

    net.setInput(inpBlob);

    Mat output = net.forward();
    ofstream file;
        
    int H = output.size[2];
    int W = output.size[3];
    string num=to_string(static_cast<long long>(f));
    file.open(txtFile+"00000"+num+".txt");
    file<<"00000"+num<<endl;
    // find the position of the body parts
    vector<Point> points(nPoints);
    for (int n=0; n < nPoints; n++)
    {
        // Probability map of corresponding body's part.
        Mat probMap(H, W, CV_32F, output.ptr(0,n));
        resize(probMap, probMap, Size(frameWidth, frameHeight));

        Point maxLoc;
        double prob;
        minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
        if (prob > thresh)
        {
            circle(frameCopy, cv::Point((int)maxLoc.x, (int)maxLoc.y), 3, Scalar(0,255,255), -1);
            cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)maxLoc.x, (int)maxLoc.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        }
        points[n] = maxLoc;
        if(n<nPoints-1){
        file<<points[n].x<<","<<points[n].y<<endl;
        }
    }

    int nPairs = sizeof(POSE_PAIRS)/sizeof(POSE_PAIRS[0]);
        
    for (int n = 0; n < nPairs; n++)
    {
        // lookup 2 connected body/hand parts
        Point2f partA = points[POSE_PAIRS[n][0]];
        Point2f partB = points[POSE_PAIRS[n][1]];

        if (partA.x<=0 || partA.y<=0 || partB.x<=0 || partB.y<=0)
            continue;

        line(frame, partA, partB, Scalar(0,255,255), 3);
        circle(frame, partA, 3, Scalar(0,0,255), -1);
        circle(frame, partB, 3, Scalar(0,0,255), -1);
       
    }
    
    file.close();
    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    cout << "Time Taken = " << t << endl;
   // imshow("Output-Keypoints", frameCopy);
  //  imshow("Output-Skeleton", frame);
        Mat res;
        frame.copyTo(res);
         
        ss.str("");
       ss << result << setw(8) << setfill('0') << f2 << ".png";
       imwrite(ss.str(), res,q);
    }
    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // part3: caculate the center point of each hand based on the key points
    get_center();  //to get the centerpoints of the original data
    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // part4: use k-means to cluster different hand gestures.
    ifstream center_reader;
  //  center_reader.open(centerFile+"centerpoints.txt");
    vector<points> dataset=openFile("/Users/actbee/Documents/Vision/final/CLASSIFY/images/centerpoints.txt");
     k_means(dataset,10);
    bool read_img=false;
    int line=1;
  //  cout<<dataset.size()<<endl;
    center_reader.open("/Users/actbee/Documents/Vision/final/CLASSIFY/images/clustering.txt");
while(line<=dataset.size()-1){
        string in_line;
        center_reader>>in_line;
        vector<string>newcomer=split(in_line,",");
        int label=atoi(newcomer[2].c_str());
        
        //now read the rect image
        ss.str("");
        int ff=100;
   while(read_img==false){
         ss.str("");
        ff+=1;
       if(ff>999){
            ss << rectFile << setw(9) << setfill('0') << ff << ".png";
        }
        else{
            ss << rectFile << setw(8) << setfill('0') << ff << ".png";
        }
        
        Mat frame = imread(ss.str());
        if(!frame.data) {
            read_img=false;
            cout <<"no image data " <<ff<< endl;
        } // break
        else{
            read_img=true;
            cout<<"have read"<<setw(8)<<setfill('0')<<f<<endl;
            Mat final_res;
            frame.copyTo(final_res);
             
            ss.str("");
            string label_file=to_string(label);
           ss << finalFile+label_file+"/"<< setw(8) << setfill('0') << ff << ".png";
           imwrite(ss.str(), final_res,q);
        }
   }
        read_img=false;
        line++;
}
    center_reader.close();
    
    return 0;
}
