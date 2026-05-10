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

/*
 * IMPORTANT:
 * This MUST be IconAnimation (not Icon)
 * because your asset is PNG frames.
 */
extern const IconAnimation A_drop_animation_128x64;

typedef struct {
    DropPhase phase;
    uint32_t phase_start_frame;
    uint32_t now_frame;
    bool cue_played;

    /* animation frame index */
    uint8_t anim_frame;
} DropModel;

/* ---------- FALLING ANIMATION ---------- */

static void drop_view_draw_falling(Canvas* canvas, DropModel* m) {
    if(!canvas || !m) return;

    /*
     * FIX:
     * correct API = canvas_draw_icon_animation
     * correct type = IconAnimation*
     */
    canvas_draw_icon_animation(
        canvas,
        0,
        0,
        (IconAnimation*)&A_drop_animation_128x64);
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
        canvas_set_font(canvas, FontSecondary);
        canvas_draw_str(canvas, 10, 30, "Press OK to start");
        break;

    case DropPhaseCountdown:
        canvas_set_font(canvas, FontPrimary);
        canvas_draw_str(canvas, 18, 10, "GET READY");

        canvas_set_font(canvas, FontBigNumbers);

        {
            uint32_t sec = (m->now_frame - m->phase_start_frame) / 30;
            char buf[8];

            if(sec < 3) {
                snprintf(buf, sizeof(buf), "%lu", (unsigned long)(3 - sec));
            } else {
                snprintf(buf, sizeof(buf), "GO");
            }

            canvas_draw_str(canvas, 50, 35, buf);
        }
        break;

    case DropPhaseFalling:
        drop_view_draw_falling(canvas, m);
        break;

    case DropPhaseDone:
        canvas_set_font(canvas, FontPrimary);
        canvas_draw_str(canvas, 25, 20, "SAMPLE DONE");
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
                m->cue_played = false;
            }

            else if(m->phase == DropPhaseDone) {
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

        /* countdown -> falling */
        if(m->phase == DropPhaseCountdown && local >= COUNTDOWN_FRAMES) {
            m->phase = DropPhaseFalling;
            m->phase_start_frame = frame;
        }

        /* falling -> done */
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