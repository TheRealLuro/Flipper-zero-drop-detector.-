#include "fidget_view.h"

#include <furi.h>
#include <gui/icon_animation.h>
#include <gui/view.h>
#include <input/input.h>

extern const Icon A_fidgeting_128x64;

typedef struct {
    IconAnimation* anim;
    uint32_t frame;
} FidgetModel;

static void fidget_view_draw(Canvas* canvas, void* model) {
    FidgetModel* m = model;

    canvas_clear(canvas);

    if(m->anim) {
        canvas_draw_icon_animation(canvas, 0, 0, m->anim);
    }

    canvas_set_font(canvas, FontSecondary);
    canvas_draw_str(canvas, 2, 62, "REC");
}

static bool fidget_view_input(InputEvent* event, void* context) {
    UNUSED(event);
    UNUSED(context);
    return false;
}

static void fidget_view_enter(void* context) {
    View* view = context;
    with_view_model(view, FidgetModel* m, {
        if(m->anim) icon_animation_start(m->anim);
    }, false);
}

static void fidget_view_exit(void* context) {
    View* view = context;
    with_view_model(view, FidgetModel* m, {
        if(m->anim) icon_animation_stop(m->anim);
    }, false);
}

void fidget_view_tick(View* view, uint32_t frame) {
    with_view_model(view, FidgetModel* m, {
        m->frame = frame;
    }, true);
}

View* fidget_view_alloc(void) {
    View* view = view_alloc();
    view_allocate_model(view, ViewModelTypeLocking, sizeof(FidgetModel));

    IconAnimation* anim = icon_animation_alloc(&A_fidgeting_128x64);
    view_tie_icon_animation(view, anim);

    with_view_model(view, FidgetModel* m, {
        m->anim = anim;
    }, false);

    view_set_draw_callback(view, fidget_view_draw);
    view_set_input_callback(view, fidget_view_input);
    view_set_enter_callback(view, fidget_view_enter);
    view_set_exit_callback(view, fidget_view_exit);

    return view;
}

void fidget_view_free(View* view) {
    IconAnimation* anim = NULL;
    with_view_model(view, FidgetModel* m, {
        anim = m->anim;
        m->anim = NULL;
    }, false);

    if(anim) {
        icon_animation_stop(anim);
        icon_animation_free(anim);
    }

    view_free(view);
}
