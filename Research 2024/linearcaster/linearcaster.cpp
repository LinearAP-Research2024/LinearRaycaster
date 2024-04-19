#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <thread>
#include <fstream>
#include <string>
#include <sstream>

struct Point3D {
    float x;
    float y;
    float z;
    
    Point3D(float x_val, float y_val, float z_val) : x(x_val), y(y_val), z(z_val) {}
};
struct Point2D {
    float x;
    float y;
    
    Point2D(float x_val, float y_val) : x(x_val), y(y_val) {}
};
struct RayPara {
    Point3D p;
    Point3D d;

    RayPara(Point3D p_val,Point3D d_val) : p(p_val), d(d_val){}
};
struct PlanePara {
    Point3D c;
    Point3D n;

    PlanePara(Point3D c_val, Point3D n_val) : c(c_val), n(n_val){}
};
struct Triangle2D {
    Point2D v1;
    Point2D v2;
    Point2D v3;
    Triangle2D(Point2D v1_val, Point2D v2_val, Point2D v3_val) : v1(v1_val),v2(v2_val),v3(v3_val){}
};
struct Triangle3D {
    Triangle2D triangle;
    PlanePara plane;
    int color;
    Triangle3D(Triangle2D t_val, PlanePara p_val, int color_val) : triangle(t_val), plane(p_val), color(color_val){}
};

float perpDotPara(
    Point2D a_1,
    Point2D b_1
){
    return a_1.x * b_1.y - a_1.y*b_1.x;
}

bool halfPlanePip(
    Point2D t_1,  
    Point2D t_2,
    Point2D t_3,
    Point2D p_1
){
    Point2D p_t1(p_1.x - t_1.x,p_1.y- t_1.y);
    Point2D p_t2(p_1.x - t_2.x,p_1.y- t_2.y);
    Point2D p_t3(p_1.x - t_3.x,p_1.y- t_3.y);

    Point2D t_12(t_1.x - t_2.x,t_1.y- t_2.y);
    Point2D t_23(t_2.x - t_3.x,t_2.y- t_3.y);
    Point2D t_31(t_3.x - t_1.x,t_3.y- t_1.y);
    auto c1 = perpDotPara(t_12, p_t2);
    auto c2 = perpDotPara(t_23, p_t3);
    auto c3 = perpDotPara(t_31, p_t1);
    //printf(":%f %f %f", c1,c2,c3);
    return (c1 > 0 && c2 > 0 && c3 > 0) || (c1 < 0 && c2 < 0 && c3 < 0);
    //return (perpDotP(t_1,p_1) > 0 && perpDotP(t_2,p_1) > 0 && perpDotP(t_3,p_1) > 0);
}

float simp_rref_for_y(
    RayPara p_1,
    PlanePara r_1
){

    float denom = 1 / (r_1.n.z * p_1.d.z + r_1.n.y * p_1.d.y + r_1.n.x * p_1.d.x);
    float numer = (r_1.n.z * p_1.d.z + r_1.n.x * p_1.d.x)*p_1.p.y
                - r_1.n.z * p_1.d.y * p_1.p.z
                - r_1.n.x * p_1.d.y * p_1.p.x
                + r_1.n.z * p_1.d.y * r_1.c.z
                + r_1.n.y * p_1.d.y * r_1.c.y
                + r_1.n.x * p_1.d.y * r_1.c.x;
    //printf(": %f ", denom * numer);
    //printf(": %f ", (r_1.n.z * p_1.d.z + r_1.n.y * p_1.d.y + r_1.n.x * p_1.d.x));
    //printf(": %f %f %f", r_1.n.z * p_1.d.z , r_1.n.z , p_1.d.z);
    return denom * numer;
}

bool ray_poly_intersect(
    RayPara p_1,
    PlanePara r_1,
    Triangle2D t_1
){
    float y_i = simp_rref_for_y(p_1, r_1);
    float inv_dy = 1 / p_1.d.y;
    float x_i = p_1.d.x * inv_dy * (y_i - p_1.p.y) + p_1.p.x;
    //printf("y: %f, x: %f" , y_i, x_i);
    Point2D p_i(x_i,y_i);
    //printf("I'm here!");
    //return halfPlanePip(t_1.v1,t_1.v2,t_1.v3,p_i);
    /*
    printf("[[%f,%f],[%f,%f],[%f,%f],[%f,%f],False],\n",
        t_1.v1.x,
        t_1.v1.y,
        t_1.v2.x,
        t_1.v2.y,
        t_1.v3.x,
        t_1.v3.y,
        p_i.x,
        p_i.y
    );*/
    if( halfPlanePip(t_1.v1,t_1.v2,t_1.v3,p_i)){
        //printf("test");
        return true;
    } else {
        return false;
    }
}
void detector_cell_pipe(
    int id,
    Point3D emitter,
    Point3D detector,
    std::vector<Triangle3D>& polygons,
    std::vector<int>& image_output
){
    Point3D direction_v(detector.x - emitter.x, detector.y - emitter.y, detector.z - emitter.z);
    RayPara cur_ray(emitter, direction_v);
    //iterates through polygons
    //printf("\n %f %f %f ",detector.x, detector.y, detector.z);
    //printf("\n %f %f %f ",direction_v.x, direction_v.y, direction_v.z);
    int hits = 0;
    int summater =0;
    //printf("[");
    for(int i = 0; i < polygons.size(); i++){
        if(ray_poly_intersect(cur_ray, polygons[i].plane, polygons[i].triangle)){
            //printf("I hit something!");
            summater += polygons[i].color;
            hits+= 1;
        }
    }
    if(hits != 0 && id < image_output.size()){
        //printf("  %d id  %d", summater / hits, id);
        image_output[id] = summater / hits;
    }
}
std::vector<int> gen_detector_cells(
    int resolution,
    Point3D emitter,
    Point3D emitter_direction,
    int bounds_size,
    std::vector<Triangle3D> polygons
){
    //printf("boys");
    std::vector<int> image_output;
    image_output.resize(resolution * resolution, 0);
    std::vector<std::thread> detectors;
    //printf("are");
    int half_res = resolution / 2;
    float bound_modifier = (static_cast< float >(bounds_size)) / resolution;
    for(int i = 0; i < resolution * resolution; i++){
        float x =  (i % resolution - half_res);
        //printf("uwu");
        float yi =  half_res - (i / resolution);
        //printf("twowu");
        float y = yi + 0.0;
        Point3D cur_detector(x * bound_modifier, y * bound_modifier, 0);
        //detectors.emplace_back(detector_cell_pipe, i, emitter, cur_detector, polygons, image_output);
        //*
        Point3D direction_v(cur_detector.x - emitter.x, cur_detector.y - emitter.y, cur_detector.z - emitter.z);
        RayPara cur_ray(emitter, direction_v);
        int hits = 0;
        int summater =0;
        for(int i = 0; i < polygons.size(); i++){
            if(ray_poly_intersect(cur_ray, polygons[i].plane, polygons[i].triangle)){
                summater += polygons[i].color;
                hits+= 1;
            }
        }
        
        if(hits != 0){
            image_output[i] = summater;
        }
        //image_output[i] = summater;
        //*/
    }
    //for (std::thread& t : detectors) {
    //    t.join();
    //}
    
    return image_output;
}
Triangle3D string_to_poly(
    const std::string& input
){
    std::vector<float> plane;
    std::vector<float> triangle;
    int index1 = input.find("=");
    int index = input.find("-");
    //printf("test1");
    int color = 0;
    //printf("%s",input.substr(0,index1));
    if(input.substr(0,1) != "*"){
        color = std::stoi(input.substr(0,index1));
    } else {
        color = -1 * std::stoi(input.substr(1,index1));
    }
    //printf("test2\n");
    std::string plane_str = input.substr(index1 + 1,index);
    std::string tri_str = input.substr(1 + index);

    std::istringstream ss(plane_str);
    // Skip the opening curly brace
    char brace;
    ss >> brace;
    // Parse floats and store in the result vector
    float num;
    while (ss >> num) {
        plane.push_back(num);

        // Skip comma and space
        char comma;
        ss >> comma;
    }
    // Skip the closing curly brace
    ss >> brace;

    std::istringstream ss2(tri_str);
    // Skip the opening curly brace
    
    ss2 >> brace;
    // Parse floats and store in the result vector
    
    while (ss2 >> num) {
        triangle.push_back(num);

        // Skip comma and space
        char comma;
        ss2 >> comma;
    }
    // Skip the closing curly brace
    ss2 >> brace;
    Point2D v1(triangle[0],triangle[1]);
    Point2D v2(triangle[2],triangle[3]);
    Point2D v3(triangle[4],triangle[5]);
    Triangle2D triangle_2(v1,v2,v3);

    Point3D n(plane[0],plane[1],plane[2]);
    Point3D c(plane[3],plane[4],plane[5]);
    PlanePara plane_2(c,n);

    Triangle3D triangle_final(triangle_2, plane_2, color);
    return triangle_final;
}
std::vector<Triangle3D> polygons;
void compile_polyimage(std::string path){
    //*
    // Open a file for reading
    std::ifstream inputFile(path);

    // Check if the file is open
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
    }
    //printf("test0");

    // Read the file into a vector of strings
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(inputFile, line)) {
        // Add each line to the vector
        //printf("%s\n", line.c_str());
        lines.push_back(line);
    }

    // Close the file
    inputFile.close();
    
    //printf("test3");
    //*/
    // Create an input string stream
    /*
    std::istringstream iss(polygon_data);

    // Vector to store separated lines
    std::vector<std::string> lines;
    
    // Read lines from the stringstream and store them in the vector
    std::string line;
    while (std::getline(iss, line)) {
        lines.push_back(line);
    }
    */
    for(int i =0; i< lines.size();i++){
    //for (const auto& str : lines) {
        //printf("%s \n", lines[i].c_str());
        polygons.push_back(string_to_poly(lines[i].c_str()));
        //printf("%d",i);
    }
}
std::vector<int> linear_cast(
    int resolution,
    Point3D emitter,
    Point3D emitter_direction,
    int bounds_size
){
    /*
    //read in polygons
    std::vector<Triangle3D> polygons;
    //*
    // Open a file for reading
    std::ifstream inputFile("polydata.txt");

    // Check if the file is open
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
    }

    // Read the file into a vector of strings
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(inputFile, line)) {
        // Add each line to the vector
        //printf("%s\n", line.c_str());
        lines.push_back(line);
    }

    // Close the file
    inputFile.close();
    // Create an input string stream
    for(int i =0; i< lines.size();i++){
    //for (const auto& str : lines) {
        //printf("%s \n", lines[i].c_str());
        polygons.push_back(string_to_poly(lines[i].c_str()));
    }
    */
    
    return gen_detector_cells(resolution, emitter, emitter_direction, bounds_size, polygons);
}
std::vector<float> tensorToList(torch::Tensor tensor) {
    // Assuming the tensor is a 1D tensor of float32 values
    int numel = tensor.numel(); // Number of elements in the tensor
    std::vector<float> resultList(numel); // Create a vector to store elements

    // Get pointer to tensor data
    float* data = tensor.data_ptr<float>();

    // Copy tensor elements to the vector
    for (int i = 0; i < numel; ++i) {
        resultList[i] = data[i];
    }

    return resultList;
}
std::vector<int> linear_wrapper(
    int resolution,
    torch::Tensor emitter,
    torch::Tensor emitter_direction,
    int bounds_size
){
    //std::vector<int> j;
    //j.resize(resolution, 0);
    //return j;
    std::vector<float> emitter_list = tensorToList(emitter);
    std::vector<float> direction_list = tensorToList(emitter_direction);
    Point3D emitter_p(emitter_list[0],emitter_list[1],emitter_list[2]);
    Point3D direction_p(direction_list[0],direction_list[1],direction_list[2]);
    
    return linear_cast(resolution, emitter_p, direction_p, bounds_size);
}

std::unordered_map<int,int> image_lookup = {};
int dda_dimension = 512;
void compile_dda_voxels(){
    std::ifstream inputFile("2levelvoxeldata.txt");

    // Check if the file is open
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
    }

    // Read the file into a vector of strings
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(inputFile, line)) {
        // Add each line to the vector
        //printf("%s\n", line.c_str());
        lines.push_back(line);
    }
    inputFile.close();
    //dda_dimension = lines[0].size();
    for(int i=0;i<lines.size();i++){
        int j = 0;
        /*
        for(char& c : lines[i]){
            image_lookup[i*dda_dimension + j] = (c=='1');
            j++;
        }*/
        if(i%(512) == 0){
            printf("%d",i / (512 * 512));}
        image_lookup[i] = stoi(lines[i]);
    }
}
int dda_traverse(
    RayPara cur_ray
){
    //GREEN BOX
    //dx,dy,dz = 1
    double X1 = cur_ray.p.x;
    double Y1 = cur_ray.p.y;
    double Z1 = cur_ray.p.z;
    
    double X2 = cur_ray.d.x;
    double Y2 = cur_ray.d.y;
    double Z2 = cur_ray.d.z;
    if(X2 - X1 == 0 || Y2 - Y1 == 0 || Z2 - Z1 == 0){
        return 0;
    }
    double xdenom = 1 / (X2 - X1);
    double ydenom = 1 / (Y2 - Y1);
    double zdenom = 1 / (Z2 - Z1);

    double a_x1 = -1 * X1 * xdenom;
    double a_y1 = -1 * Y1 * ydenom;
    double a_z1 = -1 * Z1 * zdenom;
    
    double a_xn = (dda_dimension - 1 - X1) * xdenom;
    double a_yn = (dda_dimension - 1 - Y1) * ydenom;
    double a_zn = (dda_dimension - 1 - Z1) * zdenom;


    //YELLOW BOX
    double a_min = std::max(
        std::max(
            0.0,
            std::min(a_x1, a_xn)
        ),
        std::max(
            std::min(a_y1,a_yn),
            std::min(a_z1,a_zn)
        )
    );
    double a_max = std::min(
        std::min(
            1.0,
            std::max(a_x1, a_xn)
        ),
        std::min(
            std::max(a_y1,a_yn),
            std::max(a_z1,a_zn)
        )
    );

    if(a_max <= a_min){
        return 0;
    }
    //printf("\nEmtter: %f %f %f",X1,Y1,Z1);
    //printf("\nDetector: %f %f %f",X2,Y2,Z2);
    //printf("\nx bounds: %f %f",a_x1,a_xn);
    //printf("\ny bounds: %f %f",a_y1,a_yn);
    //printf("\nz bounds: %f %f",a_z1,a_zn);
    //printf("\na bounds: %f %f",a_min,a_max);

    //ORANGE BOX
    double i_min = 0, j_min = 0, k_min = 0;
    double i_max = 0, j_max = 0, k_max = 0;
    if(X2-X1 > 0){
        i_min = dda_dimension - (dda_dimension - 1 - a_min * (X2 - X1) - X1);
        i_max = 1 + (X1 + a_max * (X2 - X1));
    } else{
        i_min = dda_dimension - (dda_dimension - 1 - a_max * (X2 - X1) - X1);
        i_max = 1 + (X1 + a_min * (X2 - X1));
    }

    if(Y2-Y1 > 0){
        j_min = dda_dimension - (dda_dimension - 1 - a_min * (Y2 - Y1) - Y1);
        j_max = 1 + (Y1 + a_max * (Y2 - Y1));
    } else{
        j_min = dda_dimension - (dda_dimension - 1 - a_max * (Y2 - Y1) - Y1);
        j_max = 1 + (Y1 + a_min * (Y2 - Y1));
    }
    
    if(Z2-Z1 > 0){
        k_min = dda_dimension - (dda_dimension - 1 - a_min * (Z2 - Z1) - Z1);
        k_max = 1 + (Z1 + a_max * (Z2 - Z1));
    } else{
        k_min = dda_dimension - (dda_dimension - 1 - a_max * (Z2 - Z1) - Z1);
        k_max = 1 + (Z1 + a_min * (Z2 - Z1));
    }
    //printf("\ni bounds: %f %f xdenom: %f",i_min,i_max,xdenom);
    //printf("\nj bounds: %f %f ydenom: %f",j_min,j_max,ydenom);
    //printf("\nk bounds: %f %f zdenom: %f",k_min,k_max,zdenom);

    //RED BOX
    std::vector<double> a_x;
    std::vector<double> a_y;
    std::vector<double> a_z;
    
    if(X2-X1>0){
        double ax_max = (i_max - 1) * xdenom;
        double a_cur = (i_min -1) * xdenom;
        while(a_cur<=ax_max){
            //printf(" %f", a_cur * (X2-X1));
            a_x.push_back(a_cur);
            a_cur += xdenom;
        }
    }else{
        double ax_min = (i_min - 1) * xdenom;
        double a_cur = (i_max -1) * xdenom;
        while(a_cur>=ax_min){
            a_x.push_back(a_cur);
            a_cur -= xdenom;
        }
    }

    if(Y2-Y1>0){
        double ay_max = (j_max - 1) * ydenom;
        double a_cur = (j_min -1) * ydenom;
        while(a_cur<=ay_max){
            a_y.push_back(a_cur);
            a_cur += ydenom;
        }
    }else{
        double ay_min = (j_min - 1) * ydenom;
        double a_cur = (j_max -1) * ydenom;
        while(a_cur>=ay_min){
            a_y.push_back(a_cur);
            a_cur -= ydenom;
        }
    }

    if(Z2-Z1>0){
        double az_max = (k_max - 1) * zdenom;
        double a_cur = (k_min -1) * zdenom;
        while(a_cur<=k_max){
            a_z.push_back(a_cur);
            a_cur += zdenom;
        }
    }else{
        double az_min = (k_min - 1) * zdenom;
        double a_cur = (k_max -1) * zdenom;
        while(a_cur>=k_min){
            a_z.push_back(a_cur);
            a_cur -= zdenom;
        }
    }


    //PURPLE BOX
    std::vector<double> a;
    int iter2 = 2 + a_x.size() + a_y.size() + a_z.size();
    if(iter2 == 0){
        return 0;
    }
    a.resize(2 + a_x.size() + a_y.size() + a_z.size());
    a[0] = a_min;
    int iter = 1;
    /*
    int ii = 0;
    int ji = 0;
    int ki = 0;
    for(int i99 = 0; i99 < iter2; i99++){
        if(ii < a_x.size()){
            if(jj < a_y.size()){
                if(kk < a_z.size()){
                    if(a_x[ii] < a_y[jj]){
                        if(a_z[kk])
                    }
                }else{

                }
            }else{
                if(kk < a_z.size()){

                }else{

                }
            }
        }else{
            if(jj < a_y.size()){
                if(kk < a_z.size()){

                }else{

                }
            }else{
                if(kk < a_z.size()){

                }else{
                    break;
                }
            }
        }
    }*/
    for(int i = 0; i < a_x.size(); i++){
        a[iter] = a_x[i];
        iter++;
    }
    for(int i = 0; i < a_y.size(); i++){
        a[iter] = a_y[i];
        iter++;
    }
    for(int i = 0; i < a_z.size(); i++){
        a[iter] = a_z[i];
        iter++;
    }
    a[iter] = a_max;
    iter++;

    //HOT PINK BOX
    int n = (int) ((i_max - i_min + 1) + (j_max - j_min + 1) + (k_max - k_min + 1) + 1);
    //printf("%d %d \n",n,iter);
    
    //WHITE BOX
    double d_1_2 = sqrt((X2 - X1) * (X2 - X1) + (Y2 - Y1) * (Y2 - Y1) + (Z2 - Z1) * (Z2 - Z1)); 
    /*
    double l_m(int m){
        return d_1_2 * (a[m] - a[m-1])
    }*/

    
    //BLACK BOX
    double summater = 0;
    int count_hits = 0;
    /*
    if(a.size() <= n){
        printf("\nERRORORRO");
        printf("\nEmtter: %f %f %f",X1,Y1,Z1);
        printf("\nDetector: %f %f %f",X2,Y2,Z2);
        printf("\niter: %d  n: %d", iter, n);
    }*/
    for(int m = 1; m<=a.size() - 1; m++){
        double a_mid = (a[m] + a[m-1]) / 2;
        double l_m = d_1_2 * (a[m] - a[m-1]);
        l_m = 1;
        //LIGHT BLUE BOX
        int i_m = int(1 + X1 + a_mid * (X2-X1));
        int j_m = int(1 + Y1 + a_mid * (Y2-Y1));
        int k_m = int(1 + Z1 + a_mid * (Z2-Z1));
        //printf("a%f",a_mid);
        //summater += ((256-i_m)*(256-i_m) + (256-j_m)*(256-j_m) + (256-k_m)*(256-k_m)) / 8000;
        
        //summater += l_m * image_lookup[int(i_m * dda_dimension * dda_dimension + j_m * dda_dimension + k_m + 0.5)];
        int hit = image_lookup[int(i_m * dda_dimension + j_m)];
        if(k_m < 11 && count_hits == 0){
            if(hit >= 0){
                summater += image_lookup[int(j_m * dda_dimension + i_m )];
                count_hits+=1;
            }
        }
    }
    if(count_hits == 0){
        return 0;
    }
    
    return int(summater);
}
std::vector<int> dda_cast(
    int resolution,
    Point3D emitter,
    Point3D emitter_direction,
    int bounds_size
){
    std::vector<int> image_output;
    image_output.resize(resolution * resolution, 0);
    std::vector<std::thread> detectors;

    //default settings:
    /*
    dda_dimension = 512;
    X,Y,Z_plane(1) = 0
    dx,dy,dz = 1

    */

    int half_res = resolution / 2;
    float bound_modifier = (static_cast< float >(bounds_size)) / resolution;
    for(int i = 0; i < resolution * resolution; i++){
        float x =  (i % resolution );

        float yi =  (i / resolution);
        float y = yi + 0.0;
        Point3D cur_detector(x * bound_modifier, y * bound_modifier, 0);
        //Point3D direction_v(cur_detector.x - emitter.x, cur_detector.y - emitter.y, cur_detector.z - emitter.z);
        RayPara cur_ray(emitter, cur_detector);
        int hits = 0;
        image_output[i] = dda_traverse(cur_ray);
    }
    //for (std::thread& t : detectors) {
    //    t.join();
    //}
    
    return image_output;
}

std::vector<int> dda_wrapper(
    int resolution,
    torch::Tensor emitter,
    torch::Tensor emitter_direction,
    int bounds_size
){
    std::vector<float> emitter_list = tensorToList(emitter);
    std::vector<float> direction_list = tensorToList(emitter_direction);
    Point3D emitter_p(emitter_list[0],emitter_list[1],emitter_list[2]);
    Point3D direction_p(direction_list[0],direction_list[1],direction_list[2]);
    
    return dda_cast(resolution, emitter_p, direction_p, bounds_size);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_raycast", &linear_wrapper, "Runs a linear raycast on the compiled polygons");
  m.def("compile_polyimage", &compile_polyimage, "Compiles polydata.txt to list");
  m.def("compile_voxelimage", &compile_dda_voxels, "Compiles ddavoxels.txt to list");
  m.def("siddon_raycast", &dda_wrapper, "Runs a dda raycast on the compiled voxels");
}