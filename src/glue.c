#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include "evalsuite.h"

static SEXP es_call(SEXP metric, SEXP args) {
    char *r = evalsuite_call(CHAR(STRING_ELT(metric, 0)), CHAR(STRING_ELT(args, 0)));
    if (r == NULL) {
        const char *e = evalsuite_last_error();
        Rf_error("%s", e ? e : "evalsuite error");
    }
    SEXP out = PROTECT(Rf_mkString(r));
    evalsuite_free(r);
    UNPROTECT(1);
    return out;
}

static SEXP es_metrics(void) {
    char *r = evalsuite_metrics();
    SEXP out = PROTECT(Rf_mkString(r ? r : "[]"));
    evalsuite_free(r);
    UNPROTECT(1);
    return out;
}

static SEXP es_version(void) { return Rf_mkString(evalsuite_version()); }

static const R_CallMethodDef methods[] = {
    {"C_es_call", (DL_FUNC)&es_call, 2},
    {"C_es_version", (DL_FUNC)&es_version, 0},
    {"C_es_metrics", (DL_FUNC)&es_metrics, 0},
    {NULL, NULL, 0}};

void R_init_evalsuite(DllInfo *dll) {
    R_registerRoutines(dll, NULL, methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
