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

/*
 * IMPORTANT:
 * This is NOT RAM-loaded animation.
 * It's a ROM icon animation (Flash-based).
 */
extern const Icon A_drop_animation_128x64;

typedef struct {
    DropPhase phase;
    uint32_t phase_start_frame;
    uint32_t now_frame;
    bool cue_played;
} DropModel;

/* ---------- UI HELPERS ---------- */

static void drop_view_draw_flipper_glyph(Canvas* canvas, uint8_t cx, uint8_t top) {
    uint8_t fx = cx - 7;
    uint8_t fy = top;

    canvas_draw_box(canvas, fx, fy, 14, 18);

    canvas_set_color(canvas, ColorWhite);
    canvas_draw_box(canvas, fx + 4, fy + 1, 6, 1);
    canvas_draw_box(canvas, fx + 1, fy + 4, 12, 5);
    canvas_draw_box(canvas, fx + 6, fy + 11, 2, 6);
    canvas_draw_box(canvas, fx + 4, fy + 13, 6, 2);
    canvas_set_color(canvas, ColorBlack);

    canvas_draw_line(canvas, fx + 3, fy + 5, fx + 9, fy + 5);
    canvas_draw_line(canvas, fx + 3, fy + 7, fx + 7, fy + 7);
}

static void drop_view_draw_drop_arrow(Canvas* canvas, uint8_t cx, uint8_t top) {
    uint8_t tip = top + 7;
    canvas_draw_line(canvas, cx, top, cx, tip);
    canvas_draw_line(canvas, cx - 1, top, cx - 1, tip - 2);
    canvas_draw_line(canvas, cx + 1, top, cx + 1, tip - 2);
    canvas_draw_line(canvas, cx - 3, tip - 3, cx, tip);
    canvas_draw_line(canvas, cx + 3, tip - 3, cx, tip);
}

/* ---------- SCREENS ---------- */

static void drop_view_draw_arming(Canvas* canvas) {
    canvas_draw_box(canvas, 0, 0, 128, 11);
    canvas_set_color(canvas, ColorWhite);
    canvas_set_font(canvas, FontPrimary);
    canvas_draw_str_aligned(canvas, 64, 2, AlignCenter, AlignTop, "DROP MODE");
    canvas_set_color(canvas, ColorBlack);

    drop_view_draw_flipper_glyph(canvas, 64, 14);
    drop_view_draw_drop_arrow(canvas, 64, 34);

    canvas_set_font(canvas, FontSecondary);
    canvas_draw_str_aligned(canvas, 64, 43, AlignCenter, AlignTop, "Press OK, then drop.");

    elements_button_center(canvas, "Begin");
    elements_button_left(canvas, "Back");
}

static void drop_view_draw_countdown(Canvas* canvas, DropModel* m) {
    uint32_t local = m->now_frame - m->phase_start_frame;
    uint32_t sec = local / 30;

    canvas_set_font(canvas, FontPrimary);
    canvas_draw_str_aligned(canvas, 64, 4, AlignCenter, AlignTop, "GET READY");

    canvas_set_font(canvas, FontBigNumbers);

    if(sec < 3) {
        char digit[4];
        snprintf(digit, sizeof(digit), "%lu", (unsigned long)(3 - sec));
        canvas_draw_str_aligned(canvas, 64, 32, AlignCenter, AlignCenter, digit);
    } else {
        canvas_set_font(canvas, FontPrimary);
        canvas_draw_str_aligned(canvas, 64, 32, AlignCenter, AlignCenter, "DROP!");
    }

    uint32_t in_sec = local % 30;
    uint8_t r = 12 + (in_sec * 12) / 30;
    if(r < 28) canvas_draw_circle(canvas, 64, 32, r);
}

/* ✅ FIXED HERE (this was your crash) */
static void drop_view_draw_falling(Canvas* canvas, DropModel* m) {
    UNUSED(m); // <-- FIX: removes compiler error cleanly

    canvas_draw_icon_animation(canvas, 0, 0, &A_drop_animation_128x64);

    canvas_draw_frame(canvas, 0, 2, 128, 3);
}

static void drop_view_draw_done(Canvas* canvas) {
    canvas_draw_box(canvas, 0, 0, 128, 11);
    canvas_set_color(canvas, ColorWhite);
    canvas_set_font(canvas, FontPrimary);
    canvas_draw_str_aligned(canvas, 64, 2, AlignCenter, AlignTop, "RECORDED");
    canvas_set_color(canvas, ColorBlack);

    canvas_set_font(canvas, FontSecondary);
    canvas_draw_frame(canvas, 14, 22, 100, 14);
    canvas_draw_str_aligned(canvas, 64, 25, AlignCenter, AlignTop, "Sample collected");

    elements_button_left(canvas, "Back");
    elements_button_center(canvas, "Again");
}

/* ---------- MAIN DRAW ---------- */

static void drop_view_draw(Canvas* canvas, void* model) {
    DropModel* m = model;

    canvas_clear(canvas);
    canvas_set_color(canvas, ColorBlack);

    switch(m->phase) {
    case DropPhaseArming:
        drop_view_draw_arming(canvas);
        break;

    case DropPhaseCountdown:
        drop_view_draw_countdown(canvas, m);
        break;

    case DropPhaseFalling:
        drop_view_draw_falling(canvas, m);
        break;

    case DropPhaseDone:
        drop_view_draw_done(canvas);
        break;
    }
}

/* ---------- INPUT ---------- */

static bool drop_view_input(InputEvent* event, void* context) {
    View* view = context;
    bool consumed = false;

    if(event->type != InputTypeShort && event->type != InputTypePress) return false;

    with_view_model(view, DropModel * m, {
        switch(m->phase) {
        case DropPhaseArming:
            if(event->key == InputKeyOk) {
                m->phase = DropPhaseCountdown;
                m->phase_start_frame = m->now_frame;
                m->cue_played = false;
                consumed = true;
            }
            break;

        case DropPhaseDone:
            if(event->key == InputKeyOk) {
                m->phase = DropPhaseArming;
                consumed = true;
            }
            break;

        default:
            break;
        }
    }, true);

    return consumed;
}

/* ---------- LIFECYCLE ---------- */

static void drop_view_enter(void* context) {
    View* view = context;

    with_view_model(view, DropModel * m, {
        m->phase = DropPhaseArming;
        m->now_frame = 0;
        m->phase_start_frame = 0;
        m->cue_played = false;
    }, true);
}

static void drop_view_exit(void* context) {
    UNUSED(context);
}

void drop_view_tick(View* view, uint32_t frame) {
    with_view_model(view, DropModel * m, {
        m->now_frame = frame;

        uint32_t local = frame - m->phase_start_frame;

        if(m->phase == DropPhaseCountdown && !m->cue_played && local >= 3 * 30) {
            m->cue_played = true;
            if(drop_notif) notification_message(drop_notif, &sequence_audiovisual_alert);
        }

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
    view_set_enter_callback(view, drop_view_enter);
    view_set_exit_callback(view, drop_view_exit);

    return view;
}

void drop_view_free(View* view) {
    view_free(view);
}