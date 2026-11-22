#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "ojasx_math_core.hpp"
using namespace ojasx::math::lattice;
template<class T> std::vector<T> read(std::ifstream& f,size_t n){std::vector<T> v(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));if(!f)throw std::runtime_error("short read");return v;}
int main(int argc,char**argv){
 for(int i=1;i<argc;i++){
  std::ifstream f(argv[i],std::ios::binary);int M,K,N,G;f.read(reinterpret_cast<char*>(&M),4);f.read(reinterpret_cast<char*>(&K),4);f.read(reinterpret_cast<char*>(&N),4);f.read(reinterpret_cast<char*>(&G),4);
  auto x=read<float>(f,M*K);size_t pw=std::string(argv[i]).find("ternary")!=std::string::npos?(size_t)N*K*3/16:(size_t)N*K/4;
  auto w=read<uint8_t>(f,pw);auto s=read<float>(f,(size_t)N*K/G);auto b=read<float>(f,N);auto dy=read<float>(f,M*N);auto expected=read<float>(f,M*N);auto expected_dx=read<float>(f,M*K);
  std::vector<float> y(M*N),dx(M*K);
  if(std::string(argv[i]).find("ternary")!=std::string::npos){fused_ternary_e8_gemm_forward<float>(x.data(),w.data(),s.data(),b.data(),y.data(),1,K,N,G);fused_ternary_e8_gemm_backward_dx<float>(dy.data(),w.data(),s.data(),dx.data(),1,K,N,G);for(int m=1;m<M;m++){fused_ternary_e8_gemm_forward<float>(x.data()+m*K,w.data(),s.data(),b.data(),y.data()+m*N,1,K,N,G);fused_ternary_e8_gemm_backward_dx<float>(dy.data()+m*N,w.data(),s.data(),dx.data()+m*K,1,K,N,G);}}
  else {fused_e8_gemm_forward<float>(x.data(),w.data(),s.data(),b.data(),y.data(),1,K,N,G);fused_e8_gemm_backward_dx<float>(dy.data(),w.data(),s.data(),dx.data(),1,K,N,G);for(int m=1;m<M;m++){fused_e8_gemm_forward<float>(x.data()+m*K,w.data(),s.data(),b.data(),y.data()+m*N,1,K,N,G);fused_e8_gemm_backward_dx<float>(dy.data()+m*N,w.data(),s.data(),dx.data()+m*K,1,K,N,G);}}
  float ey=0,edx=0;for(size_t j=0;j<y.size();j++)ey=std::max(ey,std::abs(y[j]-expected[j]));for(size_t j=0;j<dx.size();j++)edx=std::max(edx,std::abs(dx[j]-expected_dx[j]));
  std::cout<<argv[i]<<" forward_max_abs="<<ey<<" backward_dx_max_abs="<<edx<<"\n";
 }
}
