#include "walking_view.h"

#include <furi.h>
#include <gui/icon_i.h>
#include <input/input.h>   // ✅ REQUIRED for InputEvent + UNUSED macro safety

extern const Icon A_walking_animation_128x64;

typedef struct {
    uint32_t frame;
} WalkModel;

/* ---------- DRAW ---------- */

static void walking_view_draw(Canvas* canvas, void* model) {
    WalkModel* m = model;

    canvas_clear(canvas);

    canvas_draw_icon(canvas, 0, 0, &A_walking_animation_128x64);

    canvas_set_font(canvas, FontSecondary);

    char buf[24];
    uint32_t sec = m->frame / 30;

    snprintf(buf, sizeof(buf), "REC %02lu:%02lu",
             (unsigned long)(sec / 60),
             (unsigned long)(sec % 60));

    canvas_draw_str(canvas, 2, 62, buf);
}

/* ---------- INPUT ---------- */

static bool walking_view_input(InputEvent* event, void* context) {
    UNUSED(event);
    UNUSED(context);
    return false;
}

/* ---------- LIFECYCLE ---------- */

static void walking_view_enter(void* context) {
    UNUSED(context);
}

static void walking_view_exit(void* context) {
    UNUSED(context);
}

/* ---------- TICK ---------- */

void walking_view_tick(View* view, uint32_t frame) {
    with_view_model(view, WalkModel* m, {
        m->frame = frame;
    }, true);
}

/* ---------- ALLOC ---------- */

View* walking_view_alloc(void) {
    View* view = view_alloc();
    view_allocate_model(view, ViewModelTypeLocking, sizeof(WalkModel));

    view_set_draw_callback(view, walking_view_draw);
    view_set_input_callback(view, walking_view_input);
    view_set_enter_callback(view, walking_view_enter);
    view_set_exit_callback(view, walking_view_exit);

    return view;
}

/* ---------- FREE ---------- */

void walking_view_free(View* view) {
    view_free(view);
}