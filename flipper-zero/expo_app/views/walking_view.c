#include "walking_view.h"

#include <furi.h>
#include <gui/icon_i.h>
#include <input/input.h>   // ✅ REQUIRED for InputEvent + UNUSED macro safety

/*
 * Frame animation
 */
extern const Icon A_walking_animation_128x64_frame_0;
extern const Icon A_walking_animation_128x64_frame_1;
extern const Icon A_walking_animation_128x64_frame_2;
extern const Icon A_walking_animation_128x64_frame_3;
extern const Icon A_walking_animation_128x64_frame_4;
extern const Icon A_walking_animation_128x64_frame_5;
extern const Icon A_walking_animation_128x64_frame_6;
extern const Icon A_walking_animation_128x64_frame_7;
extern const Icon A_walking_animation_128x64_frame_8;
extern const Icon A_walking_animation_128x64_frame_9;
extern const Icon A_walking_animation_128x64_frame_10;
extern const Icon A_walking_animation_128x64_frame_11;
extern const Icon A_walking_animation_128x64_frame_12;
extern const Icon A_walking_animation_128x64_frame_13;
extern const Icon A_walking_animation_128x64_frame_14;
extern const Icon A_walking_animation_128x64_frame_15;

static const Icon* walking_frames[] = {
    &A_walking_animation_128x64_frame_0,
    &A_walking_animation_128x64_frame_1,
    &A_walking_animation_128x64_frame_2,
    &A_walking_animation_128x64_frame_3,
    &A_walking_animation_128x64_frame_4,
    &A_walking_animation_128x64_frame_5,
    &A_walking_animation_128x64_frame_6,
    &A_walking_animation_128x64_frame_7,
    &A_walking_animation_128x64_frame_8,
    &A_walking_animation_128x64_frame_9,
    &A_walking_animation_128x64_frame_10,
    &A_walking_animation_128x64_frame_11,
    &A_walking_animation_128x64_frame_12,
    &A_walking_animation_128x64_frame_13,
    &A_walking_animation_128x64_frame_14,
    &A_walking_animation_128x64_frame_15,
};

#define WALKING_FRAME_COUNT (sizeof(walking_frames) / sizeof(walking_frames[0]))

typedef struct {
    uint32_t frame;
} WalkModel;

/* ---------- DRAW ---------- */

static void walking_view_draw(Canvas* canvas, void* model) {
    WalkModel* m = model;

    canvas_clear(canvas);

    uint32_t idx = m->frame % WALKING_FRAME_COUNT;
    canvas_draw_icon(canvas, 0, 0, walking_frames[idx]);


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