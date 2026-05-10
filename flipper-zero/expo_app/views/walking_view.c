#include "walking_view.h"

#include <furi.h>
#include <gui/icon_animation.h>
#include <gui/view.h>
#include <input/input.h>

extern const Icon A_walking_animation_128x64;

typedef struct {
    IconAnimation* anim;
    uint32_t frame;
} WalkModel;

/* ---------- DRAW ---------- */

static void walking_view_draw(Canvas* canvas, void* model) {
    WalkModel* m = model;

    canvas_clear(canvas);

    if(m->anim) {
        canvas_draw_icon_animation(canvas, 0, 0, m->anim);
    }

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

    IconAnimation* anim = icon_animation_alloc(&A_walking_animation_128x64);
    view_tie_icon_animation(view, anim);
    icon_animation_start(anim);

    with_view_model(view, WalkModel* m, {
        m->anim = anim;
    }, false);

    view_set_draw_callback(view, walking_view_draw);
    view_set_input_callback(view, walking_view_input);
    view_set_enter_callback(view, walking_view_enter);
    view_set_exit_callback(view, walking_view_exit);

    return view;
}

/* ---------- FREE ---------- */

void walking_view_free(View* view) {
    IconAnimation* anim = NULL;
    with_view_model(view, WalkModel* m, {
        anim = m->anim;
        m->anim = NULL;
    }, false);

    if(anim) {
        icon_animation_stop(anim);
        icon_animation_free(anim);
    }

    view_free(view);
}
