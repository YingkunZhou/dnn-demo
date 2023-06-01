#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <cfloat>
#include <cstdio>

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module> repeat warmup\n";
    return -1;
  }

  int repeat_loop = argc > 2? atoi(argv[2]) : 100;
  int warmup_loop = argc > 3? atoi(argv[3]) : 10;
  bool useCUDA = argc == 5;


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    if (useCUDA)
      module = torch::jit::load(argv[1], torch::kCUDA);
    else
      module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  if (useCUDA)
    inputs.push_back(torch::rand({1, 3, 224, 224}).to(at::kCUDA));
  else
    inputs.push_back(torch::rand({1, 3, 224, 224}));

  at::Tensor output;

  for (int iter = 0; iter < warmup_loop; iter++) {
    // Execute the model and turn its output into a tensor.
    output = module.forward(inputs).toTensor().to(at::kCPU);
  }

  double time_min = DBL_MAX;
  double time_max = -DBL_MAX;
  double time_avg = 0;
  struct timespec begin, end;


  for (int iter = 0; iter < repeat_loop; iter++) {
    clock_gettime(CLOCK_REALTIME, &begin);
    output = module.forward(inputs).toTensor();
    clock_gettime(CLOCK_REALTIME, &end);
    long long seconds = end.tv_sec - begin.tv_sec;
    long long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double time = seconds*1e3 + nanoseconds * 1e-6;
    time_min = std::min(time_min, time);
    time_max = std::max(time_max, time);
    time_avg += time;
  }
  time_avg /= repeat_loop;
  fprintf(stderr, "min = %7.2f  max = %7.2f  avg = %7.2f\n", time_min, time_max, time_avg);

  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
