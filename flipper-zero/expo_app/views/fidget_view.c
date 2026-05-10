#include "fidget_view.h"

#include <furi.h>
#include <gui/icon_i.h>
#include <input/input.h>

/*
 * Frame animation
 */
extern const Icon A_fidgeting_128x64_frame_00;
extern const Icon A_fidgeting_128x64_frame_01;
extern const Icon A_fidgeting_128x64_frame_02;
extern const Icon A_fidgeting_128x64_frame_03;
extern const Icon A_fidgeting_128x64_frame_04;
extern const Icon A_fidgeting_128x64_frame_05;
extern const Icon A_fidgeting_128x64_frame_06;
extern const Icon A_fidgeting_128x64_frame_07;
extern const Icon A_fidgeting_128x64_frame_08;
extern const Icon A_fidgeting_128x64_frame_09;
extern const Icon A_fidgeting_128x64_frame_10;
extern const Icon A_fidgeting_128x64_frame_11;
extern const Icon A_fidgeting_128x64_frame_12;
extern const Icon A_fidgeting_128x64_frame_13;
extern const Icon A_fidgeting_128x64_frame_14;
extern const Icon A_fidgeting_128x64_frame_15;

static const Icon* fidget_frames[] = {
    &A_fidgeting_128x64_frame_00,
    &A_fidgeting_128x64_frame_01,
    &A_fidgeting_128x64_frame_02,
    &A_fidgeting_128x64_frame_03,
    &A_fidgeting_128x64_frame_04,
    &A_fidgeting_128x64_frame_05,
    &A_fidgeting_128x64_frame_06,
    &A_fidgeting_128x64_frame_07,
    &A_fidgeting_128x64_frame_08,
    &A_fidgeting_128x64_frame_09,
    &A_fidgeting_128x64_frame_10,
    &A_fidgeting_128x64_frame_11,
    &A_fidgeting_128x64_frame_12,
    &A_fidgeting_128x64_frame_13,
    &A_fidgeting_128x64_frame_14,
    &A_fidgeting_128x64_frame_15,
};

#define FIDGET_FRAME_COUNT (sizeof(fidget_frames) / sizeof(fidget_frames[0]))

typedef struct {
    uint32_t frame;
} FidgetModel;

static void fidget_view_draw(Canvas* canvas, void* model) {
    FidgetModel* m = model;

    canvas_clear(canvas);

    uint32_t idx = m->frame % FIDGET_FRAME_COUNT;
    canvas_draw_icon(canvas, 0, 0, fidget_frames[idx]);


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
    with_view_model(view, FidgetModel* m, {
        m->frame = frame;
    }, true);
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

void fidget_view_free(View* view) {
    view_free(view);
}