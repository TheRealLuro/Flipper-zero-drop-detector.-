#include "drop_view.h"

#include <furi.h>
#include <gui/elements.h>
#include <gui/icon_i.h>
#include <gui/canvas.h>
#include <input/input.h>
#include <notification/notification_messages.h>

static NotificationApp* drop_notif = NULL;

void drop_view_set_notification(NotificationApp* notif) {
    drop_notif = notif;
}

typedef enum {
    DropPhaseArming,
    DropPhaseCountdown,
    DropPhaseFalling,
    DropPhaseDone,
} DropPhase;

#define COUNTDOWN_FRAMES (4 * 30)
#define FALLING_FRAMES   (75)

/* =========================
   MANUAL FRAME ANIMATION
   ========================= */


extern const Icon A_drop_animation_128x64_frame_000;
extern const Icon A_drop_animation_128x64_frame_001;
extern const Icon A_drop_animation_128x64_frame_002;
extern const Icon A_drop_animation_128x64_frame_003;
extern const Icon A_drop_animation_128x64_frame_004;
extern const Icon A_drop_animation_128x64_frame_005;
extern const Icon A_drop_animation_128x64_frame_006;
extern const Icon A_drop_animation_128x64_frame_007;
extern const Icon A_drop_animation_128x64_frame_008;
extern const Icon A_drop_animation_128x64_frame_009;
extern const Icon A_drop_animation_128x64_frame_010;
extern const Icon A_drop_animation_128x64_frame_011;
extern const Icon A_drop_animation_128x64_frame_012;
extern const Icon A_drop_animation_128x64_frame_013;
extern const Icon A_drop_animation_128x64_frame_014;
extern const Icon A_drop_animation_128x64_frame_015;

static const Icon* drop_frames[] = {
    &A_drop_animation_128x64_frame_000,
    &A_drop_animation_128x64_frame_001,
    &A_drop_animation_128x64_frame_002,
    &A_drop_animation_128x64_frame_003,
    &A_drop_animation_128x64_frame_004,
    &A_drop_animation_128x64_frame_005,
    &A_drop_animation_128x64_frame_006,
    &A_drop_animation_128x64_frame_007,
    &A_drop_animation_128x64_frame_008,
    &A_drop_animation_128x64_frame_009,
    &A_drop_animation_128x64_frame_010,
    &A_drop_animation_128x64_frame_011,
    &A_drop_animation_128x64_frame_012,
    &A_drop_animation_128x64_frame_013,
    &A_drop_animation_128x64_frame_014,
    &A_drop_animation_128x64_frame_015,
};

#define DROP_FRAME_COUNT (sizeof(drop_frames) / sizeof(drop_frames[0]))

typedef struct {
    DropPhase phase;
    uint32_t phase_start_frame;
    uint32_t now_frame;

    uint8_t anim_frame;
} DropModel;

/* ---------- FALLING DRAW ---------- */

static void drop_view_draw_falling(Canvas* canvas, DropModel* m) {
    if(!m) return;

    /* advance frame */
    uint8_t frame = m->anim_frame % DROP_FRAME_COUNT;

    canvas_draw_icon(canvas, 0, 0, drop_frames[frame]);
}

/* ---------- MAIN DRAW ---------- */

static void drop_view_draw(Canvas* canvas, void* model) {
    DropModel* m = model;
    if(!m) return;

    canvas_clear(canvas);
    canvas_set_color(canvas, ColorBlack);

    switch(m->phase) {

    case DropPhaseArming:
        canvas_set_font(canvas, FontPrimary);
        canvas_draw_str(canvas, 20, 10, "DROP MODE");
        break;

    case DropPhaseCountdown:
        canvas_set_font(canvas, FontPrimary);
        canvas_draw_str(canvas, 18, 10, "READY");
        break;

    case DropPhaseFalling:
        drop_view_draw_falling(canvas, m);
        break;

    case DropPhaseDone:
        canvas_set_font(canvas, FontPrimary);
        canvas_draw_str(canvas, 25, 20, "DONE");
        break;
    }
}

/* ---------- INPUT ---------- */

static bool drop_view_input(InputEvent* event, void* context) {
    View* view = context;

    if(event->type != InputTypeShort && event->type != InputTypePress)
        return false;

    with_view_model(view, DropModel* m, {
        if(event->key != InputKeyOk) {
            return;
        }

        // OK behavior:
        // - From ARMINING -> start countdown.
        // - From DONE -> go back to ARMINING.
        // - During COUNTDOWN/FALLING -> ignore to prevent stuck phases/counters.
        if(m->phase == DropPhaseArming) {
            m->phase = DropPhaseCountdown;
            m->phase_start_frame = m->now_frame;
            m->anim_frame = 0;
        } else if(m->phase == DropPhaseDone) {
            m->phase = DropPhaseArming;
            m->phase_start_frame = m->now_frame;
            m->anim_frame = 0;
        }
    }, true);

    return true;
}

/* ---------- TICK ---------- */

void drop_view_tick(View* view, uint32_t frame) {
    if(!view) return;

    with_view_model(view, DropModel* m, {
        if(!m) return;

        m->now_frame = frame;

        uint32_t local = frame - m->phase_start_frame;

        /* countdown → falling */
        if(m->phase == DropPhaseCountdown && local >= COUNTDOWN_FRAMES) {
            m->phase = DropPhaseFalling;
            m->phase_start_frame = frame;
            m->anim_frame = 0;
        }

        /* animate frames ONLY during falling */
        if(m->phase == DropPhaseFalling) {
            m->anim_frame++;
        }

        /* falling → done */
        if(m->phase == DropPhaseFalling && local >= FALLING_FRAMES) {
            m->phase = DropPhaseDone;
        }

    }, true);
}

/* ---------- ALLOC ---------- */

View* drop_view_alloc(void) {
    View* view = view_alloc();
    if(!view) return NULL;

    view_allocate_model(view, ViewModelTypeLocking, sizeof(DropModel));

    view_set_draw_callback(view, drop_view_draw);
    view_set_input_callback(view, drop_view_input);

    return view;
}

void drop_view_free(View* view) {
    if(view) view_free(view);
}

