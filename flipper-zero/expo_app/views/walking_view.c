#include "walking_view.h"

#include <furi.h>
#include <gui/icon_i.h>
#include <input/input.h>

extern const Icon A_walking_animation_128x64;

typedef struct {
    uint32_t frame;
} WalkModel;

/* ---------- SIDEWALK ---------- */

#define SIDEWALK_TOP 48
#define SIDEWALK_BOT 55
#define JOINT_GAP   18

static void walking_draw_sidewalk(Canvas* canvas, uint32_t frame) {
    canvas_draw_line(canvas, 0, SIDEWALK_TOP, 127, SIDEWALK_TOP);
    canvas_draw_line(canvas, 0, SIDEWALK_BOT, 127, SIDEWALK_BOT);

    int shift = (int)((frame * 3) / 5) % JOINT_GAP;

    for(int8_t k = 0; k < 9; k++) {
        int jx = k * JOINT_GAP - shift;
        if(jx >= 0 && jx < 128) {
            canvas_draw_line(canvas, jx, SIDEWALK_TOP + 1, jx, SIDEWALK_BOT - 1);
        }
    }
}

/* ---------- HUD ---------- */

static void walking_view_draw_hud(Canvas* canvas, uint32_t frame) {
    canvas_set_font(canvas, FontSecondary);

    char buf[16];
    uint32_t sec = frame / 30;

    snprintf(buf, sizeof(buf), "REC %02lu:%02lu",
             (unsigned long)(sec / 60),
             (unsigned long)(sec % 60));

    canvas_draw_str(canvas, 2, 62, buf);
}

/* ---------- DRAW ---------- */

static void walking_view_draw(Canvas* canvas, void* model) {
    WalkModel* m = model;

    canvas_clear(canvas);
    canvas_set_color(canvas, ColorBlack);

    /* ✅ FIX: NO frame arrays, NO RAM expansion */
    canvas_draw_icon_animation(canvas, 0, 0, &A_walking_animation_128x64);

    walking_draw_sidewalk(canvas, m->frame);
    walking_view_draw_hud(canvas, m->frame);
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