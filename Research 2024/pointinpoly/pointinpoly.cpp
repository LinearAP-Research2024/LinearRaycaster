#include <torch/extension.h>
#include <iostream>
#include <vector>

// Function to convert tensor to a list (std::vector)
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
std::vector<float> cross_product(std::vector<float> tensor1, std::vector<float> tensor2) {
    // Check if tensors have the correct shape
    //TORCH_CHECK(tensor1.sizes() == torch::IntArrayRef({1, 3}), "Tensor 1 must have shape (1,3)");
    //TORCH_CHECK(tensor2.sizes() == torch::IntArrayRef({1, 3}), "Tensor 2 must have shape (1,3)");

    // Extract values from tensors
    float x1 = tensor1[0];
    float y1 = tensor1[1];
    float z1 = tensor1[2];
    float x2 = tensor2[0];
    float y2 = tensor2[1];
    float z2 = tensor2[2];

    // Calculate cross product
    float result_x = y1 * z2 - z1 * y2;
    float result_y = z1 * x2 - x1 * z2;
    float result_z = x1 * y2 - y1 * x2;

    // Create a new tensor to hold the result
    //torch::Tensor result = torch::zeros({1, 3}, tensor1.options());
    return {result_x,result_y,result_z};
}
bool IsPointOnSegment(std::vector<float> point, std::vector<float> lineStart, std::vector<float> lineEnd)
{
    float crossProduct = (point[1] - lineStart[1]) * (lineEnd[0] - lineStart[0]) - (point[0] - lineStart[0]) * (lineEnd[1] - lineStart[1]);
    if (fabs(crossProduct) > FLT_EPSILON){
        return false;
    }
    float dotProduct = (point[0] - lineStart[0]) * (lineEnd[0] - lineStart[0]) + (point[1] - lineStart[1]) * (lineEnd[1] - lineStart[1]);
    if (dotProduct < 0){
        return false;
    }
    float squaredLengthBA = (lineEnd[0] - lineStart[0]) * (lineEnd[0] - lineStart[0]) + (lineEnd[1] - lineStart[1]) * (lineEnd[1] - lineStart[1]);
    if (dotProduct > squaredLengthBA){
        return false;
    }
    return true;
}
// parameters are points for lines and b in the form {x,y,1}
// make sure to use homogenous coords
// AOT the first line cross later
int a = 0;
int intersect_check(
    torch::Tensor a_1,  
    torch::Tensor a_2,
    torch::Tensor b_1,
    torch::Tensor b_2
){
    /*
    auto a_l1 = tensorToList(a_1);
    auto a_l2 = tensorToList(a_2);
    auto b_l1 = tensorToList(b_1);
    auto b_l2 = tensorToList(b_2);

    auto l_1 = cross_product(a_l1, a_l2);
    auto l_2 = cross_product(b_l1, b_l2);
    auto s = cross_product(l_1, l_2);
    if(s[2] == 0){
        return false;
    } else{
        double tolerance = 1e-8; 
        //std::vector<float> i = {s[0] / s[2], s[1] / s[2]};
        auto j = s[0] / s[2];
        auto k = s[1] / s[2];
        //return (i[0] > a_l1[0] && i[0] > a_l2[0]) || (i[0] < a_l1[0] && i[0] < a_l2[0]);
        //return ((i[0] >= a_l1[0] && i[0] <= a_l2[0]) || (i[0] <= a_l1[0] && i[0] >= a_l2[0]));
        //return IsPointOnSegment({j,k}, {a_l1[0], a_l1[1]}, {a_l2[0], a_l2[1]});
        if(j >= std::min(a_l1[0], a_l2[0]) - tolerance && j <= std::max(a_l1[0], a_l2[0]) + tolerance){
            return (j >= std::min(b_l1[0], b_l2[0]) - tolerance && j <= std::max(b_l1[0], b_l2[0]) + tolerance);
        } else {
            return false;
        }
        //return (i[0] >= std::min(a_l1[0], a_l2[0]) && i[0] <= std::max(a_l1[0], a_l2[0]));
    }
    */
   a = a+1;
   return a;
}

int jordan_pip(
    torch::Tensor t_1,  
    torch::Tensor t_2,
    torch::Tensor t_3,
    torch::Tensor p_1
){
    auto p_l1 = tensorToList(p_1);
    auto p_2 = torch::tensor({0.0F,p_l1[1],1.0F});
    auto sum = 0;
    return (intersect_check(p_1,p_2,t_1,t_2) != intersect_check(p_1,p_2,t_2,t_3)) != intersect_check(p_1,p_2,t_1,t_3);

    /*
    if(intersect_check(p_1,p_2,t_1,t_2)){
        sum += 1;
    }
    if(intersect_check(p_1,p_2,t_2,t_3)){
        sum += 10;
    }
    if(intersect_check(p_1,p_2,t_1,t_3)){
        sum += 100;
    }
    return sum ;
    //*/
}


float calcInvSqRoot( float n ) {
   
   const float threehalfs = 1.5F;
   float y = n;
   
   long i = * ( long * ) &y;

   i = 0x5f3759df - ( i >> 1 );
   y = * ( float * ) &i;
   
   y = y * ( threehalfs - ( (n * 0.5F) * y * y ) );
   
   return y;
}
// Absolute error <= 6.7e-5
float acos_nvidia(float x) {
  float negate = float(x < 0);
  x = abs(x);
  float ret = -0.0187293;
  ret = ret * x;
  ret = ret + 0.0742610;
  ret = ret * x;
  ret = ret - 0.2121144;
  ret = ret * x;
  ret = ret + 1.5707288;
  ret = ret * sqrt(1.0-x);
  ret = ret - 2 * negate * ret;
  return negate * 3.14159265358979 + ret;
}
float vector_angle(
    torch::Tensor t_1,  
    torch::Tensor t_2
){
    auto t_l1 = tensorToList(t_1);
    auto t_l2 = tensorToList(t_2);
    return acos_nvidia(tensorToList(torch::dot(t_1, t_2))[0] * (calcInvSqRoot(
        (t_l1[0] * t_l1[0] + t_l1[1] * t_l1[1]) * (t_l2[0] * t_l2[0] + t_l2[1] * t_l2[1])
    )));
}

bool triangulation_pip(
    torch::Tensor tp_1,  
    torch::Tensor tp_2,
    torch::Tensor tp_3,
    torch::Tensor p_1
){
    auto t_1 = tp_1 - p_1;
    auto t_2 = tp_2 - p_1;
    auto t_3 = tp_3 - p_1;
    float vsum = fabs(vector_angle(t_1,t_2)) + fabs(vector_angle(t_2,t_3)) + fabs(vector_angle(t_3,t_1));
    //return std::rand() % 10 > 5;
    return 6.24F < vsum;
    return vsum;
}

float perpDotP(
    torch::Tensor a_1,
    torch::Tensor b_1
){
    auto a_l1 = tensorToList(a_1);
    auto b_l1 = tensorToList(b_1);
    return a_l1[0] * b_l1[1] - a_l1[1] * b_l1[0]; 
}

//precondition: t_1,t_2,t_3 are in clockwise order (if they aren't, things break)
bool half_plane_pip(
    torch::Tensor t_1,  
    torch::Tensor t_2,
    torch::Tensor t_3,
    torch::Tensor p_1
){
    /*
    auto c1 = perpDotP(t_1,p_1);
    auto c2 = perpDotP(t_2,p_1);
    auto c3 = perpDotP(t_3,p_1);
    */
    auto c1 = perpDotP(t_1 - t_2, p_1 - t_2);
    auto c2 = perpDotP(t_2 - t_3, p_1 - t_3);
    auto c3 = perpDotP(t_3 - t_1, p_1 - t_1);
    return (c1 > 0 && c2 > 0 && c3 > 0) || (c1 < 0 && c2 < 0 && c3 < 0);
    //return (perpDotP(t_1,p_1) > 0 && perpDotP(t_2,p_1) > 0 && perpDotP(t_3,p_1) > 0);
}
struct Point2D {
    float x;
    float y;
    
    Point2D(float x_val, float y_val) : x(x_val), y(y_val) {}
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
    return (c1 > 0 && c2 > 0 && c3 > 0) || (c1 < 0 && c2 < 0 && c3 < 0);
    //return (perpDotP(t_1,p_1) > 0 && perpDotP(t_2,p_1) > 0 && perpDotP(t_3,p_1) > 0);
}
bool half_plane_pip_2(
    std::vector<float> v1,
    std::vector<float> v2,
    std::vector<float> v3,
    std::vector<float> p1
){
    Point2D t_1(v1[0],v1[1]);
    Point2D t_2(v2[0],v2[1]);
    Point2D t_3(v3[0],v3[1]);
    Point2D p_1(p1[0],p1[1]);
    return halfPlanePip(t_1,t_2,t_3,p_1);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("jordan", &jordan_pip, "Jordan");
  m.def("triangulation", &triangulation_pip, "Triangulation");
  m.def("half_plane", &half_plane_pip, "Half Plane");
  m.def("half_para", &half_plane_pip_2, "new half plane");
  m.def("intersect_check", &intersect_check, "Intersect Check");
}