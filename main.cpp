/**
 * main.cpp — Minimum Line Cover (MLC) Solver
 *
 * Incremental Binary Feasibility Solver using the HiGHS C++ API.
 */

#include <highs/Highs.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ────────────────────────────────────────────────────────────────────────────
//  Helpers
// ────────────────────────────────────────────────────────────────────────────

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
               steady_clock::now().time_since_epoch())
        .count();
}

static long rss_kb() {
#ifdef __linux__
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            sscanf(line.c_str(), "VmRSS: %ld", &kb);
            return kb;
        }
    }
#endif
    return -1;
}

static void checked_fread(void* buf, std::size_t elem_size,
                          std::size_t count, FILE* fp,
                          const char* field_name) {
    std::size_t got = std::fread(buf, elem_size, count, fp);
    if (got != count) {
        std::fprintf(stderr,
                     "[FATAL] Short read on field '%s': expected %zu, got %zu\n",
                     field_name, count, got);
        std::exit(EXIT_FAILURE);
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  CSR Matrix  (read-only after load)
// ────────────────────────────────────────────────────────────────────────────

struct CsrMatrix {
    uint32_t n_points{0};
    uint32_t n_lines{0};
    uint64_t nnz{0};

    std::vector<uint64_t> indptr;   
    std::vector<uint32_t> indices;  

    bool load(const std::string& path) {
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::perror(("fopen: " + path).c_str());
            return false;
        }

        checked_fread(&n_points, sizeof(n_points), 1, fp, "n_points");
        checked_fread(&n_lines,  sizeof(n_lines),  1, fp, "n_lines");
        checked_fread(&nnz,      sizeof(nnz),      1, fp, "nnz");

        std::fprintf(stdout,
                     "[INFO] Matrix: %u points, %u lines, %llu nonzeros\n",
                     n_points, n_lines, (unsigned long long)nnz);

        indptr.resize(static_cast<std::size_t>(n_points) + 1);
        checked_fread(indptr.data(), sizeof(uint64_t),
                      indptr.size(), fp, "indptr");

        indices.resize(static_cast<std::size_t>(nnz));
        checked_fread(indices.data(), sizeof(uint32_t),
                      indices.size(), fp, "indices");

        std::fclose(fp);
        return true;
    }

    std::pair<const uint32_t*, const uint32_t*> row(uint32_t i) const {
        const uint32_t* beg = indices.data() + indptr[i];
        const uint32_t* end = indices.data() + indptr[i + 1];
        return {beg, end};
    }
};

// ────────────────────────────────────────────────────────────────────────────
//  HiGHS model helpers
// ────────────────────────────────────────────────────────────────────────────

static const char* model_status_str(HighsModelStatus s) {
    switch (s) {
        case HighsModelStatus::kOptimal:           return "OPTIMAL(FEASIBLE)";
        case HighsModelStatus::kInfeasible:        return "INFEASIBLE";
        case HighsModelStatus::kUnbounded:         return "UNBOUNDED";
        case HighsModelStatus::kObjectiveBound:    return "OBJ_BOUND";
        case HighsModelStatus::kObjectiveTarget:   return "OBJ_TARGET";
        case HighsModelStatus::kSolutionLimit:     return "SOLN_LIMIT";
        case HighsModelStatus::kTimeLimit:         return "TIME_LIMIT";
        case HighsModelStatus::kNotset:            return "NOT_SET";
        default:                                   return "OTHER";
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Main
// ────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "Usage: %s <matrix.bin> <results.csv> [L_init] [N_max]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string matrix_path(argv[1]);
    const std::string csv_path(argv[2]);
    int L_current  = (argc >= 4) ? std::atoi(argv[3]) : 1;
    int N_max_arg  = (argc >= 5) ? std::atoi(argv[4]) : -1;  

    if (L_current < 1) {
        std::fprintf(stderr, "[FATAL] L_init must be >= 1\n");
        return EXIT_FAILURE;
    }

    CsrMatrix mat;
    if (!mat.load(matrix_path)) return EXIT_FAILURE;

    const uint32_t N_total = mat.n_points;
    const uint32_t N_max   = (N_max_arg <= 0 || static_cast<uint32_t>(N_max_arg) > N_total)
                                 ? N_total
                                 : static_cast<uint32_t>(N_max_arg);
    const uint32_t n_lines = mat.n_lines;

    std::fprintf(stdout,
                 "[INFO] Solving n=1..%u  |  columns(lines)=%u  |  L_init=%d\n",
                 N_max, n_lines, L_current);

    FILE* csv_fp = std::fopen(csv_path.c_str(), "a");
    if (!csv_fp) {
        std::perror(("fopen csv: " + csv_path).c_str());
        return EXIT_FAILURE;
    }
    
    if (std::ftell(csv_fp) == 0) {
        std::fprintf(csv_fp, "n,L_n,solve_ms,status,rss_kb\n");
        std::fflush(csv_fp);
    }

    uint32_t resume_from = 0;
    {
        FILE* rr = std::fopen(csv_path.c_str(), "r");
        if (rr) {
            char buf[256];
            int  last_n = 0, last_L = 0;
            bool first  = true;
            while (std::fgets(buf, sizeof(buf), rr)) {
                if (first) { first = false; continue; }  
                int rn, rL;
                if (sscanf(buf, "%d,%d", &rn, &rL) == 2) {
                    if (rn > last_n) { last_n = rn; last_L = rL; }
                }
            }
            std::fclose(rr);
            if (last_n > 0) {
                resume_from = static_cast<uint32_t>(last_n);
                L_current   = last_L;
                std::fprintf(stdout,
                             "[RESUME] Found %d solved rows. Resuming from n=%u with L_current=%d\n",
                             last_n, resume_from + 1, L_current);
            }
        }
    }

    Highs highs;
    highs.setOptionValue("output_flag",   false);
    highs.setOptionValue("log_to_console", false);

    highs.setOptionValue("mip_max_nodes",          static_cast<int>(1e8));
    highs.setOptionValue("mip_rel_gap",            1e-6);
    highs.setOptionValue("mip_feasibility_tolerance", 1e-7);
    highs.setOptionValue("presolve", "on");

    // ── Add columns (Fix 1 applied here) ───────────────────────────────
    {
        std::vector<double> lb(n_lines, 0.0);
        std::vector<double> ub(n_lines, 1.0);
        
        HighsStatus st = highs.addVars(static_cast<int>(n_lines), lb.data(), ub.data());
        if (st != HighsStatus::kOk) {
            std::fprintf(stderr, "[FATAL] addVars failed\n");
            return EXIT_FAILURE;
        }

        // Fix 2 applied here
        std::vector<HighsVarType> vtypes(n_lines, HighsVarType::kInteger);
        highs.changeColsIntegrality(0, static_cast<int>(n_lines) - 1, vtypes.data());
    }

    const int limit_row_idx = 0;  
    {
        std::vector<int>    idx(n_lines);
        std::vector<double> val(n_lines, 1.0);
        for (uint32_t j = 0; j < n_lines; ++j) idx[j] = static_cast<int>(j);

        HighsStatus st = highs.addRow(
            -kHighsInf,                          
            static_cast<double>(L_current),      
            static_cast<int>(n_lines),
            idx.data(), val.data());
        if (st != HighsStatus::kOk) {
            std::fprintf(stderr, "[FATAL] addRow (limit row) failed\n");
            return EXIT_FAILURE;
        }
    }
    assert(highs.getNumRow() == 1);

    if (resume_from > 0) {
        std::fprintf(stdout, "[RESUME] Replaying %u coverage rows into HiGHS model...\n", resume_from);
        for (uint32_t i = 0; i < resume_from; ++i) {
            auto [beg, end] = mat.row(i);
            int  len        = static_cast<int>(end - beg);
            std::vector<int>    cidx(len);
            std::vector<double> cval(len, 1.0);
            for (int k = 0; k < len; ++k) cidx[k] = static_cast<int>(beg[k]);

            HighsStatus st = highs.addRow(1.0, kHighsInf, len, cidx.data(), cval.data());
            if (st != HighsStatus::kOk) {
                std::fprintf(stderr, "[FATAL] addRow during replay failed at i=%u\n", i);
                return EXIT_FAILURE;
            }
        }
        highs.changeRowBounds(limit_row_idx, -kHighsInf, static_cast<double>(L_current));
        std::fprintf(stdout, "[RESUME] Replay complete.\n");
    }

    std::vector<int>    row_cidx;
    std::vector<double> row_cval;
    row_cidx.reserve(1024);
    row_cval.reserve(1024);

    double total_solve_ms = 0.0;

    // ── Main incremental loop (Fix 3 applied here) ───────────────────────────
    for (uint32_t n = resume_from + 1; n <= N_max; ++n) {
        auto [beg, end] = mat.row(n - 1);
        int len = static_cast<int>(end - beg);

        bool is_infeasible = false;
        double solve_ms = 0.0;

        if (len == 0) {
            std::fprintf(stderr,
                         "[WARN] Point n=%u has zero candidate lines. Marking infeasible.\n", n);
            is_infeasible = true;
        } else {
            row_cidx.resize(len);
            row_cval.assign(len, 1.0);
            for (int k = 0; k < len; ++k) row_cidx[k] = static_cast<int>(beg[k]);

            HighsStatus st = highs.addRow(1.0, kHighsInf, len, row_cidx.data(), row_cval.data());
            if (st != HighsStatus::kOk) {
                std::fprintf(stderr, "[FATAL] addRow failed at n=%u\n", n);
                std::fclose(csv_fp);
                return EXIT_FAILURE;
            }

            double t0 = now_ms();
            HighsStatus run_st = highs.run();
            solve_ms = now_ms() - t0;
            total_solve_ms += solve_ms;

            if (run_st != HighsStatus::kOk && run_st != HighsStatus::kWarning) {
                std::fprintf(stderr, "[FATAL] highs.run() returned error at n=%u\n", n);
                std::fclose(csv_fp);
                return EXIT_FAILURE;
            }

            HighsModelStatus ms = highs.getModelStatus();

            if (ms == HighsModelStatus::kOptimal ||
                ms == HighsModelStatus::kObjectiveBound ||
                ms == HighsModelStatus::kObjectiveTarget ||
                ms == HighsModelStatus::kSolutionLimit) {
                
                long rss = rss_kb();
                std::fprintf(stdout, "[n=%6u] L=%4d | %8.2f ms | %s | RSS=%ld kB\n",
                             n, L_current, solve_ms, model_status_str(ms), rss);
                std::fprintf(csv_fp, "%u,%d,%.3f,%s,%ld\n",
                             n, L_current, solve_ms, model_status_str(ms), rss);
                std::fflush(csv_fp);

            } else if (ms == HighsModelStatus::kInfeasible) {
                is_infeasible = true;
            } else if (ms == HighsModelStatus::kTimeLimit) {
                std::fprintf(stderr, "[WARN] Time limit hit at n=%u.\n", n);
                std::fprintf(csv_fp, "%u,%d,%.3f,TIME_LIMIT,%ld\n", n, L_current, solve_ms, rss_kb());
                std::fflush(csv_fp);
            } else {
                std::fprintf(stderr, "[ERROR] Unexpected model status '%s' at n=%u.\n", model_status_str(ms), n);
                std::fprintf(csv_fp, "%u,%d,%.3f,ERROR_%s,%ld\n", n, L_current, solve_ms, model_status_str(ms), rss_kb());
                std::fflush(csv_fp);
            }
        }

        if (is_infeasible) {
            ++L_current;
            HighsStatus cs = highs.changeRowBounds(limit_row_idx, -kHighsInf, static_cast<double>(L_current));
            if (cs != HighsStatus::kOk) {
                std::fprintf(stderr, "[FATAL] changeRowBounds failed at n=%u\n", n);
                std::fclose(csv_fp);
                return EXIT_FAILURE;
            }

            long rss = rss_kb();
            std::fprintf(stdout, "[n=%6u] L=%4d | %8.2f ms | INFEASIBLE→BUMP | RSS=%ld kB\n",
                         n, L_current, solve_ms, rss);
            std::fprintf(csv_fp, "%u,%d,%.3f,INFEASIBLE_BUMP,%ld\n", n, L_current, solve_ms, rss);
            std::fflush(csv_fp);
        }
    }

    std::fprintf(stdout, "\n[DONE] Processed n=1..%u | Final L=%d | Total solve time=%.2f s\n",
                 N_max, L_current, total_solve_ms / 1000.0);

    std::fclose(csv_fp);
    return EXIT_SUCCESS;
}
