#include "idle_view.h"

#include <furi.h>
#include <gui/icon.h>
#include <input/input.h>

extern const Icon A_idle_128x64;

typedef struct {
    uint32_t frame;
} IdleModel;

#define SRC_STEP 5u // 30/6 -> 6fps source playback

static void idle_view_draw_hud(Canvas* canvas, uint32_t frame) {
    canvas_set_font(canvas, FontSecondary);
    char buf[24];
    uint32_t sec = frame / 30;
    snprintf(buf, sizeof(buf), "REC %02lu:%02lu",
             (unsigned long)(sec / 60), (unsigned long)(sec % 60));
    canvas_draw_str(canvas, 4, 8, buf);
    if((frame / 15) % 2 == 0) canvas_draw_disc(canvas, 122, 5, 2); // blink
}

static void idle_view_draw(Canvas* canvas, void* model) {
    IdleModel* m = model;
    canvas_clear(canvas);
    canvas_set_color(canvas, ColorBlack);

    const uint8_t fc = A_idle_128x64.frame_count;
    uint32_t src = (m->frame / SRC_STEP) % (fc ? fc : 1);
    canvas_draw_bitmap(canvas, 0, 0, 128, 64, A_idle_128x64.frames[src]);

    idle_view_draw_hud(canvas, m->frame);
}

static bool idle_view_input(InputEvent* event, void* context) {
    UNUSED(event);
    UNUSED(context);
    return false;
}

static void idle_view_enter(void* context) {
    UNUSED(context);
    // DATA: collector start
}

static void idle_view_exit(void* context) {
    UNUSED(context);
    // DATA: collector stop
}

void idle_view_tick(View* view, uint32_t frame) {
    with_view_model(view, IdleModel * m, { m->frame = frame; }, true);
}

View* idle_view_alloc(void) {
    View* view = view_alloc();
    view_allocate_model(view, ViewModelTypeLocking, sizeof(IdleModel));
    view_set_draw_callback(view, idle_view_draw);
    view_set_input_callback(view, idle_view_input);
    view_set_enter_callback(view, idle_view_enter);
    view_set_exit_callback(view, idle_view_exit);
    return view;
}

void idle_view_free(View* view) {
    view_free(view);
}
