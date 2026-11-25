#include <stdio.h>
#include "evalsuite.h"

int main(void) {
    char *r = evalsuite_call("f1", "{\"y_true\":[0,1,1,0],\"y_pred\":[0,1,0,0]}");
    if (!r) { fprintf(stderr, "error: %s\n", evalsuite_last_error()); return 1; }
    printf("%s\n", r);
    evalsuite_free(r);
    if (evalsuite_call("nope", "{}") == NULL) printf("error: %s\n", evalsuite_last_error());
    printf("version %s\n", evalsuite_version());
    return 0;
}
