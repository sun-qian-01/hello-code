#include <stdio.h>

int main() {
    int *ptr = NULL;
    printf("Attempting to dereference NULL...\n");
    int value = *ptr;   // This line will crash
    printf("Value: %d\n", value);
    return 0;
}