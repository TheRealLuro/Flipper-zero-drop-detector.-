#pragma once

#include <gui/view.h>

View* idle_view_alloc(void);
void idle_view_free(View* view);
void idle_view_tick(View* view, uint32_t frame);
