#include "branch_predictor.h"
#include <stdio.h>

#define TABLE_BITS      14
#define HIST_LEN        14
#define PERC_TABLES     8
#define PERC_HIST       62
#define PERC_ENTRIES    512
#define TRACE_LEN       2000000

int main(void)
{
    printf("branch-predictor-sim: bimodal / gshare / tournament / perceptron\n\n");
    printf("  trace length    : %d instructions\n\n", TRACE_LEN);

    BranchTrace *trace = trace_generate(TRACE_LEN, 0xdeadbeefcafe1234ull);

    BranchPredictor *bim  = bimodal_create(TABLE_BITS);
    BranchPredictor *gsh  = gshare_create(TABLE_BITS, HIST_LEN);
    BranchPredictor *tour = tournament_create(TABLE_BITS,
                                bimodal_create(TABLE_BITS),
                                gshare_create(TABLE_BITS, HIST_LEN));
    BranchPredictor *perc = perceptron_create(PERC_TABLES, PERC_HIST, PERC_ENTRIES);

    BranchPredictor *predictors[] = { bim, gsh, tour, perc };
    size_t n = sizeof(predictors) / sizeof(*predictors);

    for (size_t i = 0; i < n; ++i) {
        PredictorResult r = bp_evaluate(predictors[i], trace);
        bp_print_result(&r);
    }

    for (size_t i = 0; i < n; ++i) bp_destroy(predictors[i]);
    trace_destroy(trace);
    return 0;
}
