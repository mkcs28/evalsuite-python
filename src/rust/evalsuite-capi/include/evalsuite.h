/* EvalSuite C API. Link with -levalsuite_c (libevalsuite_c.so / .dylib / evalsuite_c.dll, or the static library). */
#ifndef EVALSUITE_H
#define EVALSUITE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Computes `metric` on JSON arguments, e.g.
 *   evalsuite_call("f1", "{\"y_true\":[0,1,1,0],\"y_pred\":[0,1,0,0]}")
 * returns "{\"metric\":\"f1\",\"value\":0.6666666666666666}". Free the result with evalsuite_free.
 * Undefined or infinite results are the JSON strings "NaN", "Infinity" and "-Infinity".
 * Add "ci":"analytic" or "ci":"bootstrap" (with "level", "n_resamples", "seed") for ci_low / ci_high.
 * Returns NULL on error; evalsuite_last_error() then gives the message. */
char *evalsuite_call(const char *metric, const char *args_json);

/* Message of the last failed call on this thread, or NULL. Owned by the library. */
const char *evalsuite_last_error(void);

/* Frees a string returned by evalsuite_call. */
void evalsuite_free(char *s);

/* Metric names as a JSON array, e.g. ["accuracy","f1",...]. Free with evalsuite_free. */
char *evalsuite_metrics(void);

/* Engine version, e.g. "1.0.0". */
const char *evalsuite_version(void);

#ifdef __cplusplus
}
#endif
#endif
