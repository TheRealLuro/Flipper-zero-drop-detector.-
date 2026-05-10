#include "drop_view.h"

#include <furi.h>
#include <gui/elements.h>
#include <gui/icon_i.h>
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

extern const Icon A_drop_animation_128x64;

typedef struct {
    DropPhase phase;
    uint32_t phase_start_frame;
    uint32_t now_frame;
    bool cue_played;
} DropModel;

/* ---------- SIMPLE DRAW ---------- */

static void drop_view_draw_falling(Canvas* canvas, DropModel* m) {
    UNUSED(m);

    /* FIX: correct API for Icon (NOT animation) */
    canvas_draw_icon(canvas, 0, 0, &A_drop_animation_128x64);
}

/* ---------- MAIN DRAW ---------- */

static void drop_view_draw(Canvas* canvas, void* model) {
    DropModel* m = model;

    canvas_clear(canvas);
    canvas_set_color(canvas, ColorBlack);

    switch(m->phase) {
    case DropPhaseArming:
        canvas_set_font(canvas, FontPrimary);
        canvas_draw_str(canvas, 20, 10, "DROP MODE");
        break;

    case DropPhaseCountdown:
        canvas_set_font(canvas, FontPrimary);
        canvas_draw_str(canvas, 20, 10, "COUNTDOWN");
        break;

    case DropPhaseFalling:
        drop_view_draw_falling(canvas, m);
        break;

    case DropPhaseDone:
        canvas_set_font(canvas, FontPrimary);
        canvas_draw_str(canvas, 20, 10, "DONE");
        break;
    }
}

/* ---------- INPUT ---------- */

static bool drop_view_input(InputEvent* event, void* context) {
    View* view = context;

    if(event->type != InputTypeShort && event->type != InputTypePress)
        return false;

    with_view_model(view, DropModel* m, {
        if(event->key == InputKeyOk) {
            if(m->phase == DropPhaseArming) {
                m->phase = DropPhaseCountdown;
                m->phase_start_frame = m->now_frame;
            } else if(m->phase == DropPhaseDone) {
                m->phase = DropPhaseArming;
            }
        }
    }, true);

    return true;
}

/* ---------- TICK ---------- */

void drop_view_tick(View* view, uint32_t frame) {
    with_view_model(view, DropModel* m, {
        m->now_frame = frame;

        uint32_t local = frame - m->phase_start_frame;

        if(m->phase == DropPhaseCountdown && local >= COUNTDOWN_FRAMES) {
            m->phase = DropPhaseFalling;
            m->phase_start_frame = frame;
        }

        if(m->phase == DropPhaseFalling && local >= FALLING_FRAMES) {
            m->phase = DropPhaseDone;
        }
    }, true);
}

/* ---------- ALLOC ---------- */

View* drop_view_alloc(void) {
    View* view = view_alloc();
    view_allocate_model(view, ViewModelTypeLocking, sizeof(DropModel));

    view_set_draw_callback(view, drop_view_draw);
    view_set_input_callback(view, drop_view_input);

    return view;
}

void drop_view_free(View* view) {
    view_free(view);
}