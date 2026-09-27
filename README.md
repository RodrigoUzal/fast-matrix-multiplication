# Fast Matrix Multiplication: Strassen vs. Standard

Advanced Algorithms, Programming Assignment 1 (Track A, empirical study).
Rodrigo Uzal Sequeros.

C++17 implementations of standard matrix multiplication (`i-k-j` loop order) and
Strassen's algorithm with a run-time cut-off, an experiment harness that times
and checks both, and a Python script that produces every figure in the report.

## Repository layout

| File | Contents |
|---|---|
| `matrix.hpp` | `Matrix` type (row-major `std::vector<double>`) and helpers: `fill_random`, `add`, `subtract`, `submatrix`, `write_block` |
| `algorithms.hpp`, `algorithms.cpp` | `standard_mul(A, B)` and `strassen_mul(A, B, cutoff)` |
| `main.cpp` | Experiment harness: generates the matrices, times both algorithms, checks every result, writes the CSV |
| `plot_graph.py` | Reads the CSV, saves the figures to `figures/`, prints the numbers quoted in the report |
| `build.bat` | Compiles with the exact flags used for the report and runs a quick smoke test |
| `results.csv`, `results_large.csv` | The measurements used in the report |
| `analysis.txt` | The numbers `plot_graph.py` printed for those measurements |
| `figures/` | The six figures in the report |

Build outputs (`*.exe`, `*.obj`) and CSV files are ignored by git, except the
two data files above, which were added on purpose.

## Requirements

- **C++:** the Microsoft C++ compiler (MSVC, from Visual Studio 2022 or the Build
  Tools), used from a *Developer PowerShell* or *Developer Command Prompt* so that
  `cl` is on the path. Any other C++17 compiler also works (see below).
- **Python 3** with `pandas`, `numpy` and `matplotlib` (3.4 or newer):
```
  python -m pip install pandas numpy matplotlib
```

## Build

From a Developer PowerShell in the repository folder:

```powershell
.\build.bat
```

This runs:

```bat
cl /nologo /std:c++17 /O2 /EHsc main.cpp algorithms.cpp /Fe:strassen.exe && strassen.exe quick
```

`/O2` turns on optimisation (every timing in the report uses it), `/EHsc`
enables standard C++ exception handling, and `/Fe:` names the executable. If
compilation succeeds, the quick smoke test runs (about one second).

With GCC or Clang instead:

```bash
g++ -std=c++17 -O2 main.cpp algorithms.cpp -o strassen
```

## Run the experiments

| Command | Sizes | Repetitions | Output | Time on the report's machine |
|---|---|---|---|---|
| `.\strassen.exe` | n = 64 to 2048, cut-off sweep at 512, 1024, 2048 | 5 | `results.csv` (165 rows) | about 8 min |
| `.\strassen.exe large` | n = 4096 | 3 | `results_large.csv` (6 rows) | about 7 min, ~1.2 GB of memory |
| `.\strassen.exe quick` | n = 64 to 512 | 1 | `results_quick.csv` (14 rows) | about 1 s |

Running `strassen.exe` or `strassen.exe large` overwrites the committed data
files. For stable timings, keep the laptop plugged in with the power mode on
best performance, close other programs, stop it from sleeping, and do not open
the CSV while the program is writing to it.

The program prints one progress line per measurement. It checks every Strassen
result against the standard one and stops if the relative discrepancy exceeds
1e-9. Matrices are seeded with `10000 * run + n`, so a rerun multiplies exactly
the same matrices; the timings, of course, depend on the machine.

## CSV format

One row per timed multiplication:

```
algorithm,n,cutoff,run,seconds,max_diff,ref_max
```

- `cutoff` is 0 for the standard algorithm, where it does not apply.
- `max_diff` is the largest |C_strassen − C_standard| (0 for the standard
  algorithm, which is the reference), and `ref_max` the largest |C_standard|.
  Their quotient is the relative discrepancy E used in the report.

## Produce the figures and numbers

In PowerShell:

```powershell
python plot_graph.py | Out-File -Encoding utf8 analysis.txt
```

This reads `results.csv`, plus `results_large.csv` if it exists, saves six PDFs
to `figures/`, and writes the numbers quoted in the report to `analysis.txt`. To
try the script on the smoke-test data instead: `python plot_graph.py results_quick.csv`.

| Figure | Objective in the report |
|---|---|
| `o1_time_vs_n.pdf`, `o1_speedup.pdf` | O1: crossover |
| `o2_cutoff_sweep.pdf` | O2: optimal cut-off |
| `o3_local_exponents.pdf` | O3: growth exponents |
| `o4_accuracy.pdf`, `o4_accuracy_vs_cutoff.pdf` | O4: accuracy |

## Using the algorithms in other code

```cpp
#include "algorithms.hpp"

std::mt19937 rng(42);
Matrix A(512, 512), B(512, 512);
fill_random(A, rng);
fill_random(B, rng);
Matrix C = strassen_mul(A, B, 64);  // cut-off 64
Matrix D = standard_mul(A, B);
```

`standard_mul` accepts any compatible sizes. `strassen_mul` requires square
matrices of the same power-of-two size and throws `std::invalid_argument`
otherwise, or if the cut-off is below 1.

## AI use

Code written with AI assistance is marked `(AI-assisted)` in its comments.
Section 4 of the report describes the use of AI tools in detail.