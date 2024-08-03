#include <stdio.h>

struct Point {
    int x;
    int y;
};

typedef struct Point Point;

void Point_init(Point * self, int x, int y) {
    self->x = x;
    self->y = y;
}

void Point_print(Point * self) {
    printf("Point(%d, %d)\n", self->x, self->y);
}

int main() {
    Point * p = Point(){};
    Point_init(&p, 10, 20);
    Point_print(&p);
    return 0;
}
