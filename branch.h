#ifndef BRANCH_PREDICTOR_H
#define BRANCH_PREDICTOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct BranchPredictor BranchPredictor;

typedef struct {
    bool  (*predict)(BranchPredictor *self, uint64_t pc);
    void  (*update) (BranchPredictor *self, uint64_t pc, bool outcome);
    void  (*destroy)(BranchPredictor *self);
    const char *name;
} BranchPredictorVtable;

struct BranchPredictor {
    const BranchPredictorVtable *vtable;
};

static inline bool bp_predict(BranchPredictor *bp, uint64_t pc)
{ return bp->vtable->predict(bp, pc); }

static inline void bp_update(BranchPredictor *bp, uint64_t pc, bool outcome)
{ bp->vtable->update(bp, pc, outcome); }

static inline void bp_destroy(BranchPredictor *bp)
{ bp->vtable->destroy(bp); }

static inline const char *bp_name(const BranchPredictor *bp)
{ return bp->vtable->name; }

BranchPredictor *bimodal_create(int table_bits);
BranchPredictor *gshare_create(int table_bits, int history_len);
BranchPredictor *tournament_create(int meta_bits,
                                   BranchPredictor *primary,
                                   BranchPredictor *secondary);
BranchPredictor *perceptron_create(int num_tables, int history_len,
                                   int table_entries);

typedef struct {
    uint64_t *pcs;
    bool     *outcomes;
    size_t    length;
} BranchTrace;

BranchTrace *trace_generate(size_t length, uint64_t seed);
void         trace_destroy(BranchTrace *t);

typedef struct {
    const char *predictor_name;
    size_t      mispredictions;
    double      mpki;
} PredictorResult;

PredictorResult bp_evaluate(BranchPredictor *bp, const BranchTrace *trace);
void            bp_print_result(const PredictorResult *r);

#endif
