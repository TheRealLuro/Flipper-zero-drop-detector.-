#include "fidget_view.h"

#include <furi.h>
#include <gui/icon_i.h>

extern const Icon A_fidgeting_128x64;

typedef struct {
    uint32_t frame;
} FidgetModel;

static void fidget_view_draw(Canvas* canvas, void* model) {
    FidgetModel* m = model;
    (void)m;

    canvas_clear(canvas);

    canvas_draw_icon(canvas, 0, 0, &A_fidgeting_128x64);

    canvas_set_font(canvas, FontSecondary);
    canvas_draw_str(canvas, 2, 62, "REC");
}

static bool fidget_view_input(InputEvent* event, void* context) {
    UNUSED(event);
    UNUSED(context);
    return false;
}

static void fidget_view_enter(void* context) { UNUSED(context); }
static void fidget_view_exit(void* context) { UNUSED(context); }

void fidget_view_tick(View* view, uint32_t frame) {
    with_view_model(view, FidgetModel* m, { m->frame = frame; }, true);
}

View* fidget_view_alloc(void) {
    View* view = view_alloc();
    view_allocate_model(view, ViewModelTypeLocking, sizeof(FidgetModel));
    view_set_draw_callback(view, fidget_view_draw);
    view_set_input_callback(view, fidget_view_input);
    view_set_enter_callback(view, fidget_view_enter);
    view_set_exit_callback(view, fidget_view_exit);
    return view;
}

void fidget_view_free(View* view) { view_free(view); }