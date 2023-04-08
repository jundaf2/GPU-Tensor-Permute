#define CATCH_CONFIG_MAIN
#include "common/catch.hpp"
#define CATCH_CONFIG_CPP11_TO_STRING
#define CATCH_CONFIG_COLOUR_ANSI
#include "seqdata_permute.cuh"


template <typename T>
class test_SeqData{

public:
  test_SeqData(std::array<int,4> dimA, std::array<int,4> permuteA){
    this->dimA = dimA;
    this->permuteA = permuteA;

    data_len = dimA[0]*dimA[1]*dimA[2]*dimA[3];
    input = std::vector<float>(data_len);
    output = std::vector<float>(data_len, 0.0f);

    
    // allocate memory for h_input and h_output
    h_input = malloc(data_len*sizeof(T));
    h_output = malloc(data_len*sizeof(T));
  }
  ~test_SeqData(){
    free(h_input);
    free(h_output);
  }

  void init_data() {

    std::mt19937 rng = std::mt19937(2023);
    std::uniform_real_distribution<float> uf_distribution = std::uniform_real_distribution<float>(-10, 10);

    std::generate(std::begin(input), std::end(input), [&]{return uf_distribution(rng);} ); 

    // copy the input to h_input data by casting float to type T
    std::transform(std::begin(input), std::end(input), reinterpret_cast<T*>(h_input), [](float x){return static_cast<T>(x);});
  }

  
  void run_gpu_permute() {
    // deivce ptr
    T *d_input, *d_output;
    
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_input, data_len*sizeof(T)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_output, data_len*sizeof(T)));

    // copy h2d
    CHECK_CUDA_ERR(cudaMemcpy(d_input,h_input,data_len*sizeof(T),cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA_ERR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    LaunchTransformSeqDataAxesKernel<T>(stream, dimA.data(), permuteA.data(), d_input, d_output);
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream));


    // copy d2h
    CHECK_CUDA_ERR(cudaMemcpy(h_output,d_output,data_len*sizeof(T),cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(cudaFree(d_input));
    CHECK_CUDA_ERR(cudaFree(d_output));
  }

  void run_cpu_permute() {

    
    std::function<int(int)> dim_factorial = [this,&dim_factorial](int n) -> int {
      assert(n<=3);
      if (n == 3) return 1;
      return dimA[n+1] * dim_factorial(n + 1);
    };

    std::function<int(int)> perm_dim_factorial = [this,&perm_dim_factorial](int n) -> int {
      assert(n<=3);
      if (n == 3) return 1;
      int np = std::find(permuteA.begin(), permuteA.end(), n+1) - permuteA.begin();
      return dimA[np] * perm_dim_factorial(permuteA[np]);
    };
    
    int stride_0 = dim_factorial(0);
    int stride_1 = dim_factorial(1);
    int stride_2 = dim_factorial(2);

    int perm_stride_0 = perm_dim_factorial(permuteA[0]);
    int perm_stride_1 = perm_dim_factorial(permuteA[1]);
    int perm_stride_2 = perm_dim_factorial(permuteA[2]);

    // cpu implementation
    for (int n = 0; n < dimA[0]; n++) {
      for (int c = 0; c < dimA[1]; c++) {
        for (int h = 0; h < dimA[2]; h++) {
          for (int w = 0; w < dimA[3]; w++) {
            int in_idx = n * stride_0 + c * stride_1 + h * stride_2 + w;
            int out_idx = n * perm_stride_0 + c * perm_stride_1 + h * perm_stride_2 + w;
            output[out_idx] = input[in_idx];
          }
        }
      }
    }
  }


  void print_vec(const std::vector<float> outv, std::string outn, int start = 0) {
      std::cout << outn << ": ";
      std::copy(outv.begin() + start, outv.begin() + ((start + 64)>outv.size()?outv.size():(start + 64)), std::ostream_iterator<float>(std::cout, ", "));
      std::cout << std::endl;
    }

  void verify() {
      // print_vec(input, "input");
      // print_vec(output, "output");
      SECTION(std::string(ANSI_COLOR_RED) + std::to_string(data_len) + ANSI_COLOR_RESET) {
        bool is_near2 = true;
        size_t count = 0;
        for (int i = 0; i < output.size(); i++) {
          bool is_this_near2 = NEAR2(static_cast<float>(reinterpret_cast<T*>(h_output)[i]), output[i], 1e-2);
          if(!is_this_near2 && count<64){
            count++;
            fmt::print(ANSI_COLOR_RED "ERROR @ {}[{}] {} vs {}\n" ANSI_COLOR_RESET, std::to_string(data_len), i, static_cast<float>(reinterpret_cast<T*>(h_output)[i]), output[i]);
          }
          is_near2 &= is_this_near2;
        }
        CHECK(is_near2);
      }
    }


private:
  std::array<int,4> dimA, permuteA;
  size_t data_len;
  std::vector<float> input, output;
  void *h_input, *h_output;

  std::function<bool(float,float,float)> NEAR2 = [](float a, float b, float prec) -> bool { return ((a != a && b != b) 
      || (a == std::numeric_limits<typename std::remove_reference<decltype(a)>::type>::infinity() 
        && b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
      || (-a == std::numeric_limits<typename std::remove_reference< decltype(a)>::type>::infinity() 
        && -b == std::numeric_limits<typename std::remove_reference<  decltype(b)>::type>::infinity()) 
      || (abs(a - b) / abs(a) < prec) || (abs(a - b) / abs(b) < prec) || (abs(a - b) < prec)); };

};

template <typename T>
int eval_seqdata(const std::array<int,4>& dim_a, const std::array<int,4>& permute_a){
  test_SeqData<T> test_seqdata(dim_a, permute_a);
  test_seqdata.init_data();
  test_seqdata.run_gpu_permute();
  test_seqdata.run_cpu_permute();
  test_seqdata.verify();
}

TEST_CASE("SeqData", "[SeqData]") {
  SECTION("1") {
    eval_seqdata<float>({16,32,64,512},{2,1,0,3});
  }
  SECTION("2") {
    eval_seqdata<__half>({16,32,64,512},{2,1,0,3});
  }

  SECTION("3") {
    eval_seqdata<float>({16,32,64,512},{2,0,1,3});
  }
  SECTION("4") {
    eval_seqdata<__half>({16,32,64,512},{2,0,1,3});
  }
  
  SECTION("5") {
    eval_seqdata<float>({16,32,64,512},{0,2,1,3});
  }
  SECTION("6") {
    eval_seqdata<__half>({16,32,64,512},{0,2,1,3});
  }

  SECTION("7") {
    eval_seqdata<float>({16,32,64,512},{1,2,0,3});
  }
  SECTION("8") {
    eval_seqdata<__half>({16,32,64,512},{1,2,0,3});
  }

  SECTION("9") {
    eval_seqdata<float>({16,32,64,512},{1,0,2,3});
  }
  SECTION("10") {
    eval_seqdata<__half>({16,32,64,512},{1,0,2,3});
  }
}