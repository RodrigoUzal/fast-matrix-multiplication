// Experiment harness for the empirical study (report, section 2).
//
// (AI-assisted) This file was written with Claude (Anthropic), following the test design we agreed
// on. It contains no algorithm code: it generates the matrices, times standard_mul and strassen_mul,
// checks every Strassen result against the standard one, and writes one CSV row per measurement.
//
// Usage, after building:
//   strassen.exe          main run:   n = 64..2048, 5 repetitions  -> results.csv
//   strassen.exe large    optional:   n = 4096,     3 repetitions  -> results_large.csv
//   strassen.exe quick    smoke test: n = 64..512,  1 repetition   -> results_quick.csv
//
// CSV columns (one row per timed multiplication):
//   algorithm  "standard" or "strassen"
//   n          matrix size (the matrices are n x n)
//   cutoff     Strassen's cut-off; 0 for standard, where it does not apply
//   run        repetition number, starting at 1
//   seconds    wall-clock time of the multiplication call only
//   max_diff   largest |C_strassen(i,j) - C_standard(i,j)|; 0 for standard (it is the reference)
//   ref_max    largest |C_standard(i,j)|, so the relative error is max_diff / ref_max
//
// Each objective of the report (Section 2.1) is answered by a filter over this single table, so no
// configuration is measured twice:
//   O1 crossover:  standard rows, plus strassen rows with cutoff == 64
//   O2 cut-off:    strassen rows with n in {512, 1024, 2048}
//   O3 exponents:  the same rows as O1, fitted over the sizes stated in Section 2.2
//   O4 accuracy:   strassen rows with cutoff == 64; E(n) in equation (1) is max_diff / ref_max
 
#include <algorithm> // std::max, std::find
#include <chrono>    // clocks
#include <cmath>     // std::fabs
#include <fstream>   // file streams
#include <iomanip>   // std::setprecision, std::setw
#include <iostream>
#include <random>
#include <string>
#include <vector>
 
#include "matrix.hpp"
#include "algorithms.hpp"
 
// steady_clock only ever moves forward at a constant rate. system_clock can jump when Windows
// synchronises the time, which would corrupt a measurement that happens to span the jump.
using Clock = std::chrono::steady_clock;
 
// ---------------------------------------------------------------------------------------------
// Experiment configuration: every number the study depends on is defined here.
// ---------------------------------------------------------------------------------------------
 
// O1, O3, O4: Strassen's cut-off while n varies.
constexpr int FIXED_CUTOFF = 64;
 
// O2: sizes at which the cut-off is swept, over every power of two from SWEEP_MIN_CUTOFF up to n.
// Only powers of two are distinct: n is a power of two and halves at every level of recursion,
// so the base case fires at the largest power of two <= cutoff (a cut-off of 100 behaves exactly
// like 64). cutoff == n means zero levels of recursion: that is the control run, whose time must
// match standard_mul's.
const std::vector<int> SWEEP_SIZES = {512, 1024, 2048};
constexpr int SWEEP_MIN_CUTOFF = 8;
 
// Largest acceptable relative difference, max_diff / ref_max. Correct runs give around 1e-14,
// so only a real bug gets anywhere near this; if it happens the program stops, because timing a
// wrong algorithm is meaningless.
constexpr double MAX_RELATIVE_ERROR = 1e-9;
 
// A plan is what one execution of the program measures: which sizes, how many repetitions, and
// which file the rows go to.
struct Plan {
    std::string csv_path;
    std::vector<int> sizes;
    int repetitions;
};
 
const Plan MAIN_PLAN  = {"results.csv",       {64, 128, 256, 512, 1024, 2048}, 5};
const Plan LARGE_PLAN = {"results_large.csv", {4096},                          3};
const Plan QUICK_PLAN = {"results_quick.csv", {64, 128, 256, 512},             1};
 
// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------
 
// The cut-offs at which Strassen is measured for size n. (Standard is always measured once too.)
static std::vector<int> cutoffs_for(int n){
    bool swept = std::find(SWEEP_SIZES.begin(), SWEEP_SIZES.end(), n) != SWEEP_SIZES.end();
    if(!swept){
        return {FIXED_CUTOFF}; // fixed cut-off only (O1, O3, O4)
    }
    std::vector<int> cutoffs;
    for(int c = SWEEP_MIN_CUTOFF; c <= n; c *= 2){
        cutoffs.push_back(c); // 8, 16, ..., n. This includes FIXED_CUTOFF, so O1 reuses that row.
    }
    return cutoffs;
}
 
// Seed for the matrices of one (run, n) group: different for every group, but the same every
// time the program runs, so any result can be reproduced. n < 10000, so run * 10000 + n never
// gives the same seed for two different groups.
static unsigned seed_for(int run, int n){
    return static_cast<unsigned>(run) * 10000u + static_cast<unsigned>(n);
}
 
// Largest |M(i,j)|.
static double max_abs(const Matrix& M){
    double largest = 0.0;
    for(double x : M.data){
        largest = std::max(largest, std::fabs(x));
    }
    return largest;
}
 
// Largest |X(i,j) - Y(i,j)|. Both matrices have the same size here, so comparing the flat
// data arrays position by position compares the same (i, j) entries.
static double max_abs_diff(const Matrix& X, const Matrix& Y){
    double largest = 0.0;
    for(std::size_t i = 0; i < X.data.size(); i++){
        largest = std::max(largest, std::fabs(X.data[i] - Y.data[i]));
    }
    return largest;
}
 
static double elapsed_seconds(Clock::time_point start, Clock::time_point end){
    return std::chrono::duration<double>(end - start).count();
}
 
// How many rows a plan will write: per size, one standard row plus one per Strassen cut-off.
static int count_measurements(const Plan& plan){
    int per_run = 0;
    for(int n : plan.sizes){
        per_run += 1 + static_cast<int>(cutoffs_for(n).size());
    }
    return per_run * plan.repetitions;
}
 
static void write_row(std::ofstream& csv, const std::string& algorithm, int n, int cutoff,
                      int run, double seconds, double max_diff, double ref_max){
    // No spaces after the commas: pandas would make them part of the column names and values.
    csv << algorithm << ',' << n << ',' << cutoff << ',' << run << ','
        << seconds << ',' << max_diff << ',' << ref_max << '\n';
    // Write the row to disk now rather than when the buffer fills up, so that if the program
    // crashes later (e.g. out of memory at n = 4096) the rows already measured are kept.
    csv.flush();
}
 
// One line per measurement on the console, so you can see where a long run is.
static void print_progress(int done, int total, int run, int n, const std::string& algorithm,
                           int cutoff, double seconds){
    std::cout << '[' << std::setw(3) << done << '/' << total << "]  run " << run
              << "  n=" << std::setw(4) << n << "  " << algorithm << "  cutoff=" << std::setw(4);
    if(algorithm == "strassen"){
        std::cout << cutoff;
    } else {
        std::cout << '-';
    }
    // std::endl (rather than '\n') also flushes: if you redirect the output to a file, e.g.
    // `strassen.exe > log.txt`, each line then appears as soon as it is printed.
    std::cout << "  " << std::fixed << std::setprecision(4) << seconds << " s" << std::endl;
}
 
// Warm-up, not recorded. The first multiplications a process performs run slower: the CPU is
// still raising its clock speed and the memory allocator is still requesting pages from Windows.
// Without this, the first row of the CSV would be measured under different conditions from the
// rest. It also checks that Strassen agrees with standard before minutes are spent measuring.
static bool warm_up(){
    const int n = 512;
    std::mt19937 rng(0);
    Matrix A(n, n);
    Matrix B(n, n);
    fill_random(A, rng);
    fill_random(B, rng);
    Matrix C_std = standard_mul(A, B);
    Matrix C_str = strassen_mul(A, B, FIXED_CUTOFF);
    double relative = max_abs_diff(C_str, C_std) / max_abs(C_std);
    std::cout << "Warm-up at n=" << n << ": relative difference " << relative << '\n';
    return relative <= MAX_RELATIVE_ERROR;
}
 
// ---------------------------------------------------------------------------------------------
// The experiment loop
// ---------------------------------------------------------------------------------------------
 
static int run_plan(const Plan& plan){
    std::ofstream csv(plan.csv_path); // creates the file, or overwrites it if it exists
    if(!csv){
        std::cerr << "Could not open " << plan.csv_path << " for writing (is it open in Excel?)\n";
        return 1;
    }
    csv << std::setprecision(10); // 10 significant digits instead of the default 6
    csv << "algorithm,n,cutoff,run,seconds,max_diff,ref_max\n"; // header row
 
    const int total = count_measurements(plan);
    std::cout << "Writing " << plan.csv_path << ": " << total << " measurements\n";
 
    if(!warm_up()){
        std::cerr << "Warm-up: Strassen disagrees with standard, stopping before measuring.\n";
        return 1;
    }
 
    int done = 0;
    const Clock::time_point experiment_start = Clock::now();
 
    // Repetitions are the OUTER loop: each run sweeps every configuration once, so a slow period
    // (thermal throttling, a background process) is spread across all configurations instead of
    // landing on the last ones. Taking the median over runs then discards it.
    for(int run = 1; run <= plan.repetitions; run++){
        for(int n : plan.sizes){
            // 1. This group's matrices (untimed). Every measurement at this (run, n) uses them.
            std::mt19937 rng(seed_for(run, n));
            Matrix A(n, n);
            Matrix B(n, n);
            fill_random(A, rng);
            fill_random(B, rng);
 
            // 2. Standard first: it is a measurement for O1 and also the reference result that
            //    every Strassen run below is compared against, so no extra multiplication is needed.
            Clock::time_point start = Clock::now();
            Matrix C_ref = standard_mul(A, B);
            Clock::time_point end = Clock::now();
            double seconds = elapsed_seconds(start, end);
            const double ref_max = max_abs(C_ref);
            write_row(csv, "standard", n, 0, run, seconds, 0.0, ref_max);
            print_progress(++done, total, run, n, "standard", 0, seconds);
 
            // 3. Strassen at every cut-off assigned to this n.
            for(int cutoff : cutoffs_for(n)){
                start = Clock::now();
                Matrix C = strassen_mul(A, B, cutoff);
                end = Clock::now();
                seconds = elapsed_seconds(start, end);
 
                // Computed after the timer stops. Because this reads every entry of C, the
                // result is used, so the compiler cannot skip the multiplication as dead code.
                const double max_diff = max_abs_diff(C, C_ref);
                write_row(csv, "strassen", n, cutoff, run, seconds, max_diff, ref_max);
                print_progress(++done, total, run, n, "strassen", cutoff, seconds);
 
                if(max_diff > MAX_RELATIVE_ERROR * ref_max){
                    std::cerr << "Strassen result is wrong at n=" << n << ", cutoff=" << cutoff
                              << " (max_diff " << max_diff << "), stopping.\n";
                    return 1;
                }
            }
        } // A, B, C_ref are destroyed here, before the next size allocates its own
    }
 
    std::cout << "Finished in " << elapsed_seconds(experiment_start, Clock::now()) << " s\n";
    return 0;
}
 
int main(int argc, char* argv[]){
    // argv[0] is the program's own name; argv[1], if present, is the first word typed after it.
    const std::string mode = (argc > 1) ? argv[1] : "main";
 
    try{
        if(mode == "main")  return run_plan(MAIN_PLAN);
        if(mode == "large") return run_plan(LARGE_PLAN);
        if(mode == "quick") return run_plan(QUICK_PLAN);
    } catch(const std::exception& e){
        // For example std::bad_alloc if n = 4096 runs out of memory. Rows already written stay.
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
 
    std::cerr << "Unknown mode '" << mode << "'. Use: strassen.exe [main | large | quick]\n";
    return 1;
}