# CliquesRS
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Rust implementations of the Bron-Kerbosch algorithm 

Current implementations:
+ Original version
+ Version using pivot
+ Version using degeneracy ordering
+ Iterative versions of the three above
+ Parallel versions of the three recursive implementations (top parallelism using [rayon](https://github.com/rayon-rs/rayon))
+ Parallel versions of the three recursive implementations (full parallelism using [threadpool](https://github.com/rust-threadpool/rust-threadpool))
+ Distributed versions of the two first recursive implementations (mpi using [rsmpi](https://github.com/rsmpi/rsmpi))

# Usage
```
Find maximal cliques using different algorithms

USAGE:
    cliques-rs [FLAGS] [OPTIONS]

FLAGS:
    -h, --help       Prints help information
    -v, --verbose    
    -V, --version    Prints version information

OPTIONS:
    -a, --algorithm <alg>...      Choose which algorithm to run [possible values: BK-Basic, BK-BasicWithPivot, BK-
                                  BasicWithOrdering, BK-BasicIterative, BK-BasicIterWithPivot, BK-
                                  BasicIterWithOrdering, BK-BasicParallel, BK-BasicParWithPivot, BK-
                                  BasicParWithOrdering, BK-BasicPool, BK-BasicPoolWithPivot]
    -i, --input-file <FILE>       Set input-file. Use STDIN if absent
    -k, --max-cliques <number>    Search maximum cliques of specific size
    -t, --threads <number>        Choose number of threads (> 0) for compatible algorithms (Defaults to number of
                                  processor cores)

```

Input data should be a list of edges separated by tabulations:
```
NODE_ID\tOTHER_NODE_ID
```

# Datasets

Some of the provided datasets were taken from the Stanford Network Analysis Project:

Jure Leskovec, and Andrej Krevl. (2014). SNAP Datasets: Stanford Large Network Dataset Collection. http://snap.stanford.edu/data.



# License

Copyright (c) 2020 Hugo Cousin

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

