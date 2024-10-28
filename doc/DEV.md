# DEV notes

## Requirements

- g++ >= 13
- clang-format
- cuda
- make

## `launch.json`

Set `"program"` with the name of the executable

## `Makefile`

Set `EXE` with the name of the executable

## CLI

Instantiate the CLI with the application description

```c++
CLI cli = CLI{"CUDA Benchmark"};
```

Declare the options

```c++
    cli
        .option({"v", "vecadd", OPTION_STRING_UNSET, "Vector addition"})
        .option({"m", "matmul", OPTION_INT_UNSET, "Matrix multiplication"})
        ...
```

Parsing the arguments marks the corresponding options as set, and replaces their default value if provided

```c++
    cli.parse(argc, argv);
```

Get a ref to the option

```c++
 auto vecadd = cli.get("vecadd");
```

Check if it was provided in the command line

```c++

        if (vecadd.isSet())
        { ...
```

Get its value (returns the default one if no value was provided)

```c++
auto size = vecadd.getValue<std::string>();
```

## Register a program

Programs are registered inside the devices loop to be ran once per device

```c++
...
// get the program option
auto vecadd = cli.get("vecadd");
// check if it was passed
if (vecadd.isSet())
{
    // get its parameter
    auto size = vecadd.getValue<std::string>();
    // start the program and OR the result
    result |= programs::vecadd(std::move(size), device, blocksize);
    // finally, continue to switch device
    continue;
}
...
```

It's essential to continue otherwise other programs will run if provided, this is not the intended behavior.

## Declare a program

Programs must be declared inside the programs namespace.
\
They generally accept a parameter (usually the dimensionality of the problem), the device to be executed on, and the blocksize option ref (to get the blocksize in case of GPU, the option is passed so the program can check if it was set).

```c++
namespace programs
{
    int vecadd(std::string&& sizeStr, Device device, Option& blocksizeOpt);
}
```

Inside the program one should process the parameter dimensionality, and the blocksize (in case of GPU execution), then make calls to the associated operation (eg. try all the dimensionalities if a dim was not provided).

## Declare an operation

Generally the operation is a void that accepts again the same problem dimensionality and the device to be executed on, but this time the numeric blocksize (at least default) in case device is GPU.
\
\
At this point the operation contains just the code to instantiate the operands with the provided dimensionality and differentiate the algorithm based on the device.
\
\
It should be enclosed inside a disable optimizations block to prevent the compiler from changing the expected performance.
\
\
It must be declared inside the operations namespace for consistency.

```c++
#pragma GCC push_options
#pragma GCC optimize("O0")

namespace operations
{
    void myop(type_t problemDim, Device device, size_t gpuThreadsPerBlock);
}

#pragma GCC pop_options
```

Example

```c++
 const std::vector<float> a = generate_random_vector(size), b = generate_random_vector(size);
        std::vector<float> c(size);

        if (device == GPU)
        {
            ScopedTimer execution(std::format("vecadd | GPU | {} [{}]", size, gpuThreadsPerBlock), PRE);
            cuda::vecadd(a.data(), b.data(), c.data(), size, gpuThreadsPerBlock);
        }
        else
        {
            ScopedTimer execution(std::format("vecadd | CPU | {}", size), PRE);
            cpu::vecadd(a, b, c, size);
        }
    }
```

The actual device algorithm should be declared, for consistency, inside the respective cpu of cuda namespace.