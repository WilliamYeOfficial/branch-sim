#include "branch_predictor.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static inline uint8_t sat_inc(uint8_t v, uint8_t max) { return v < max ? v + 1 : v; }
static inline uint8_t sat_dec(uint8_t v)               { return v > 0   ? v - 1 : v; }
static inline bool    counter_taken(uint8_t v)         { return v >= 2; }

typedef struct {
    BranchPredictor base;
    int             bits;
    uint8_t        *table;
} BimodalPredictor;

static bool bimodal_predict(BranchPredictor *self, uint64_t pc)
{
    BimodalPredictor *b = (BimodalPredictor *)self;
    return counter_taken(b->table[pc & (size_t)((1 << b->bits) - 1)]);
}

static void bimodal_update(BranchPredictor *self, uint64_t pc, bool outcome)
{
    BimodalPredictor *b = (BimodalPredictor *)self;
    uint8_t *e = &b->table[pc & (size_t)((1 << b->bits) - 1)];
    *e = outcome ? sat_inc(*e, 3) : sat_dec(*e);
}

static void bimodal_destroy(BranchPredictor *self)
{
    BimodalPredictor *b = (BimodalPredictor *)self;
    free(b->table);
    free(b);
}

static const BranchPredictorVtable bimodal_vtable = {
    bimodal_predict, bimodal_update, bimodal_destroy, "Bimodal"
};

BranchPredictor *bimodal_create(int table_bits)
{
    BimodalPredictor *b = malloc(sizeof *b);
    b->base.vtable = &bimodal_vtable;
    b->bits        = table_bits;
    b->table       = calloc((size_t)(1 << table_bits), 1);
    return &b->base;
}

typedef struct {
    BranchPredictor  base;
    int              bits;
    int              hist_len;
    uint8_t         *table;
    uint32_t         ghr;
} GSharePredictor;

static uint32_t gshare_index(GSharePredictor *g, uint64_t pc)
{
    return ((uint32_t)(pc ^ g->ghr)) & (uint32_t)((1 << g->bits) - 1);
}

static bool gshare_predict(BranchPredictor *self, uint64_t pc)
{
    GSharePredictor *g = (GSharePredictor *)self;
    return counter_taken(g->table[gshare_index(g, pc)]);
}

static void gshare_update(BranchPredictor *self, uint64_t pc, bool outcome)
{
    GSharePredictor *g = (GSharePredictor *)self;
    uint8_t *e = &g->table[gshare_index(g, pc)];
    *e    = outcome ? sat_inc(*e, 3) : sat_dec(*e);
    g->ghr = ((g->ghr << 1) | (uint32_t)outcome) & (uint32_t)((1 << g->hist_len) - 1);
}

static void gshare_destroy(BranchPredictor *self)
{
    GSharePredictor *g = (GSharePredictor *)self;
    free(g->table);
    free(g);
}

static const BranchPredictorVtable gshare_vtable = {
    gshare_predict, gshare_update, gshare_destroy, "GShare"
};

BranchPredictor *gshare_create(int table_bits, int history_len)
{
    GSharePredictor *g = malloc(sizeof *g);
    g->base.vtable = &gshare_vtable;
    g->bits        = table_bits;
    g->hist_len    = history_len;
    g->table       = calloc((size_t)(1 << table_bits), 1);
    g->ghr         = 0;
    return &g->base;
}

typedef struct {
    BranchPredictor  base;
    int              bits;
    uint8_t         *choice;
    BranchPredictor *primary;
    BranchPredictor *secondary;
} TournamentPredictor;

static bool tournament_predict(BranchPredictor *self, uint64_t pc)
{
    TournamentPredictor *t = (TournamentPredictor *)self;
    uint32_t idx = (uint32_t)(pc) & (uint32_t)((1 << t->bits) - 1);
    return counter_taken(t->choice[idx])
        ? bp_predict(t->secondary, pc)
        : bp_predict(t->primary,   pc);
}

static void tournament_update(BranchPredictor *self, uint64_t pc, bool outcome)
{
    TournamentPredictor *t = (TournamentPredictor *)self;
    bool pp = bp_predict(t->primary,   pc);
    bool sp = bp_predict(t->secondary, pc);
    uint32_t idx = (uint32_t)(pc) & (uint32_t)((1 << t->bits) - 1);
    uint8_t *c   = &t->choice[idx];

    if (pp != sp)
        *c = (sp == outcome) ? sat_inc(*c, 3) : sat_dec(*c);

    bp_update(t->primary,   pc, outcome);
    bp_update(t->secondary, pc, outcome);
}

static void tournament_destroy(BranchPredictor *self)
{
    TournamentPredictor *t = (TournamentPredictor *)self;
    free(t->choice);
    free(t);
}

static const BranchPredictorVtable tournament_vtable = {
    tournament_predict, tournament_update, tournament_destroy, "Tournament"
};

BranchPredictor *tournament_create(int meta_bits,
                                   BranchPredictor *primary,
                                   BranchPredictor *secondary)
{
    TournamentPredictor *t = malloc(sizeof *t);
    t->base.vtable = &tournament_vtable;
    t->bits        = meta_bits;
    t->choice      = calloc((size_t)(1 << meta_bits), 1);
    t->primary     = primary;
    t->secondary   = secondary;
    return &t->base;
}

#define PERC_WEIGHT_MAX 127
#define PERC_WEIGHT_MIN (-127)

typedef struct {
    BranchPredictor  base;
    int              num_tables;
    int              hist_len;
    int              entries;
    int              theta;
    int16_t        **weights;
    uint64_t        *ghr;
    int              ghr_words;
} PerceptronPredictor;

static bool ghr_get_bit(const PerceptronPredictor *p, int i)
{
    return (bool)((p->ghr[i / 64] >> (i % 64)) & 1);
}

static void ghr_push_bit(PerceptronPredictor *p, bool b)
{
    for (int w = p->ghr_words - 1; w > 0; --w)
        p->ghr[w] = (p->ghr[w] << 1) | (p->ghr[w - 1] >> 63);
    p->ghr[0] = (p->ghr[0] << 1) | (uint64_t)b;
}

static int32_t perceptron_dot(PerceptronPredictor *p, uint64_t pc)
{
    int32_t y = 0;
    int stride = p->hist_len / p->num_tables;
    for (int h = 0; h < p->num_tables; ++h) {
        uint32_t idx = (uint32_t)((pc >> (h * 3)) ^ (uint64_t)h * 2654435761ul)
                       & (uint32_t)(p->entries - 1);
        int sign = ghr_get_bit(p, h * stride) ? 1 : -1;
        y += sign * p->weights[h][idx];
    }
    return y;
}

static bool perceptron_predict(BranchPredictor *self, uint64_t pc)
{
    return perceptron_dot((PerceptronPredictor *)self, pc) >= 0;
}

static void perceptron_update(BranchPredictor *self, uint64_t pc, bool outcome)
{
    PerceptronPredictor *p = (PerceptronPredictor *)self;
    int32_t y    = perceptron_dot(p, pc);
    bool    pred = (y >= 0);
    int stride   = p->hist_len / p->num_tables;

    if (pred != outcome || abs(y) <= p->theta) {
        int t_sign = outcome ? 1 : -1;
        for (int h = 0; h < p->num_tables; ++h) {
            uint32_t idx = (uint32_t)((pc >> (h * 3)) ^ (uint64_t)h * 2654435761ul)
                           & (uint32_t)(p->entries - 1);
            int      hs  = ghr_get_bit(p, h * stride) ? 1 : -1;
            int16_t  nw  = (int16_t)(p->weights[h][idx] + t_sign * hs);
            p->weights[h][idx] = nw >  PERC_WEIGHT_MAX ?  PERC_WEIGHT_MAX
                                : nw < PERC_WEIGHT_MIN ? PERC_WEIGHT_MIN : nw;
        }
    }
    ghr_push_bit(p, outcome);
}

static void perceptron_destroy(BranchPredictor *self)
{
    PerceptronPredictor *p = (PerceptronPredictor *)self;
    for (int h = 0; h < p->num_tables; ++h) free(p->weights[h]);
    free(p->weights);
    free(p->ghr);
    free(p);
}

static const BranchPredictorVtable perceptron_vtable = {
    perceptron_predict, perceptron_update, perceptron_destroy, "Perceptron"
};

BranchPredictor *perceptron_create(int num_tables, int history_len, int table_entries)
{
    PerceptronPredictor *p = malloc(sizeof *p);
    p->base.vtable  = &perceptron_vtable;
    p->num_tables   = num_tables;
    p->hist_len     = history_len;
    p->entries      = table_entries;
    p->theta        = (int)(1.93 * history_len + 14);
    p->ghr_words    = history_len / 64 + 1;
    p->ghr          = calloc((size_t)p->ghr_words, sizeof(uint64_t));
    p->weights      = malloc(sizeof(int16_t *) * (size_t)num_tables);
    for (int h = 0; h < num_tables; ++h)
        p->weights[h] = calloc((size_t)table_entries, sizeof(int16_t));
    return &p->base;
}

static uint64_t xorshift64(uint64_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return *state;
}

BranchTrace *trace_generate(size_t length, uint64_t seed)
{
    BranchTrace *t = malloc(sizeof *t);
    t->pcs      = malloc(sizeof(uint64_t) * length);
    t->outcomes = malloc(sizeof(bool)     * length);
    t->length   = length;

    uint64_t state = seed;

    uint64_t sites[64];
    for (int i = 0; i < 64; ++i)
        sites[i] = (xorshift64(&state) & ~0xFFFull) | 0x400000ul;

    uint32_t loop_ctr[16] = {0};

    for (size_t i = 0; i < length; ++i) {
        int site   = (int)(xorshift64(&state) % 64);
        t->pcs[i]  = sites[site];
        double r   = (double)(xorshift64(&state) & 0xFFFFFF) / (double)0xFFFFFF;

        if (site < 51) {
            t->outcomes[i] = r < 0.90;
        } else if (site < 61) {
            int lc         = site - 51;
            t->outcomes[i] = (loop_ctr[lc] < 8);
            loop_ctr[lc]   = t->outcomes[i] ? loop_ctr[lc] + 1 : 0;
        } else {
            t->outcomes[i] = r < 0.52;
        }
    }
    return t;
}

void trace_destroy(BranchTrace *t)
{
    if (t) { free(t->pcs); free(t->outcomes); free(t); }
}

PredictorResult bp_evaluate(BranchPredictor *bp, const BranchTrace *trace)
{
    size_t misses = 0;
    for (size_t i = 0; i < trace->length; ++i) {
        if (bp_predict(bp, trace->pcs[i]) != trace->outcomes[i]) misses++;
        bp_update(bp, trace->pcs[i], trace->outcomes[i]);
    }
    return (PredictorResult){
        .predictor_name  = bp_name(bp),
        .mispredictions  = misses,
        .mpki            = (double)misses / ((double)trace->length / 1000.0),
    };
}

void bp_print_result(const PredictorResult *r)
{
    printf("  %-20s MPKI = %.2f\n", r->predictor_name, r->mpki);
}
