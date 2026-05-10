#include "fidget_view.h"

#include <furi.h>
#include <gui/icon_i.h>
#include <input/input.h>

extern const Icon A_fidgeting_128x64;

typedef struct {
    uint32_t frame;
} FidgetModel;

#define SRC_STEP 5u // 30/6 -> 6fps source playback

static void fidget_view_draw_hud(Canvas* canvas, uint32_t frame) {
    canvas_set_font(canvas, FontSecondary);
    canvas_draw_str(canvas, 2, 62, "REC");
    if((frame / 15) % 2 == 0) canvas_draw_disc(canvas, 22, 60, 1); // blink

    char buf[16];
    uint32_t sec_total = frame / 30;
    uint32_t mm = sec_total / 60;
    uint32_t ss = sec_total % 60;
    uint32_t tenths = (frame * 10 / 30) % 10;
    snprintf(buf, sizeof(buf), "%02lu:%02lu.%lu",
             (unsigned long)mm, (unsigned long)ss, (unsigned long)tenths);
    canvas_draw_str_aligned(canvas, 126, 62, AlignRight, AlignBottom, buf);
}

static void fidget_view_draw(Canvas* canvas, void* model) {
    FidgetModel* m = model;
    canvas_clear(canvas);
    canvas_set_color(canvas, ColorBlack);

    const uint8_t fc = A_fidgeting_128x64.frame_count;
    uint32_t src = (m->frame / SRC_STEP) % (fc ? fc : 1);
    canvas_draw_bitmap(canvas, 0, 0, 128, 64, A_fidgeting_128x64.frames[src]);

    fidget_view_draw_hud(canvas, m->frame);
}

static bool fidget_view_input(InputEvent* event, void* context) {
    UNUSED(event);
    UNUSED(context);
    return false;
}

static void fidget_view_enter(void* context) {
    UNUSED(context);
    // DATA: collector start
}

static void fidget_view_exit(void* context) {
    UNUSED(context);
    // DATA: collector stop
}

void fidget_view_tick(View* view, uint32_t frame) {
    with_view_model(view, FidgetModel * m, { m->frame = frame; }, true);
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
