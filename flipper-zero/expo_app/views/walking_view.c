#include "walking_view.h"

#include <furi.h>
#include <gui/icon.h>
#include <input/input.h>

extern const Icon A_walking_animation_128x64;

typedef struct {
    uint32_t frame;
} WalkModel;

#define SRC_STEP 5u // 30/6 -> 6fps source playback

#define SIDEWALK_TOP 48
#define SIDEWALK_BOT 55
#define JOINT_GAP   18

static void walking_draw_sidewalk(Canvas* canvas, uint32_t frame) {
    canvas_draw_line(canvas, 0, SIDEWALK_TOP, 127, SIDEWALK_TOP);
    canvas_draw_line(canvas, 0, SIDEWALK_BOT, 127, SIDEWALK_BOT);

    // expansion joints, scroll left
    int shift = (int)((frame * 3) / 5) % JOINT_GAP;
    for(int8_t k = 0; k < 9; k++) {
        int jx = k * JOINT_GAP - shift;
        if(jx < 0 || jx >= 128) continue;
        canvas_draw_line(canvas, jx, SIDEWALK_TOP + 1, jx, SIDEWALK_BOT - 1);
    }
}

static void walking_view_draw_hud(Canvas* canvas, uint32_t frame) {
    canvas_set_font(canvas, FontSecondary);
    char buf[16];
    uint32_t sec = frame / 30;
    snprintf(buf, sizeof(buf), "REC %02lu:%02lu",
             (unsigned long)(sec / 60), (unsigned long)(sec % 60));
    canvas_draw_str(canvas, 2, 62, buf);
}

static void walking_view_draw(Canvas* canvas, void* model) {
    WalkModel* m = model;
    canvas_clear(canvas);
    canvas_set_color(canvas, ColorBlack);

    const uint8_t fc = A_walking_animation_128x64.frame_count;
    uint32_t src = (m->frame / SRC_STEP) % (fc ? fc : 1);
    canvas_draw_bitmap(canvas, 0, 0, 128, 64, A_walking_animation_128x64.frames[src]);

    walking_draw_sidewalk(canvas, m->frame);
    walking_view_draw_hud(canvas, m->frame);
}

static bool walking_view_input(InputEvent* event, void* context) {
    UNUSED(event);
    UNUSED(context);
    return false;
}

static void walking_view_enter(void* context) {
    UNUSED(context);
    // DATA: collector start
}

static void walking_view_exit(void* context) {
    UNUSED(context);
    // DATA: collector stop
}

void walking_view_tick(View* view, uint32_t frame) {
    with_view_model(view, WalkModel * m, { m->frame = frame; }, true);
}

View* walking_view_alloc(void) {
    View* view = view_alloc();
    view_allocate_model(view, ViewModelTypeLocking, sizeof(WalkModel));
    view_set_draw_callback(view, walking_view_draw);
    view_set_input_callback(view, walking_view_input);
    view_set_enter_callback(view, walking_view_enter);
    view_set_exit_callback(view, walking_view_exit);
    return view;
}

void walking_view_free(View* view) {
    view_free(view);
}
