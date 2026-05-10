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

#define COUNTDOWN_FRAMES (4 * 30) // 4s @ 30fps
#define FALLING_FRAMES   (75)     // 2.5s @ 30fps

// ufbt asset, baked in by fap_icon_assets="images". forward-decl so we don't
// depend on the generated header path.
extern const Icon A_drop_animation_128x64;

typedef struct {
    DropPhase phase;
    uint32_t phase_start_frame;
    uint32_t now_frame;
    bool cue_played; // one-shot for the "DROP!" beep
} DropModel;

// 14x18 stylized flipper - black body, light screen window, dpad cross
static void drop_view_draw_flipper_glyph(Canvas* canvas, uint8_t cx, uint8_t top) {
    uint8_t fx = cx - 7;
    uint8_t fy = top;

    canvas_draw_box(canvas, fx, fy, 14, 18);

    canvas_set_color(canvas, ColorWhite);
    canvas_draw_box(canvas, fx + 4, fy + 1, 6, 1);    // speaker slit
    canvas_draw_box(canvas, fx + 1, fy + 4, 12, 5);   // screen
    canvas_draw_box(canvas, fx + 6, fy + 11, 2, 6);   // dpad |
    canvas_draw_box(canvas, fx + 4, fy + 13, 6, 2);   // dpad -
    canvas_set_color(canvas, ColorBlack);

    // screen scanlines
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

// arming page - title strip, glyph, arrow, instruction, buttons
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
    uint32_t sec = local / 30; // 0..3 (3,2,1,DROP)

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

    // pulse ring expands within each second
    uint32_t in_sec = local % 30;
    uint8_t r = 12 + (in_sec * 12) / 30;
    if(r < 28) canvas_draw_circle(canvas, 64, 32, r);

    // tick marks
    for(uint8_t i = 0; i < 3; i++) {
        uint8_t cx = 36 + i * 28;
        if(i < sec) canvas_draw_disc(canvas, cx, 58, 3);
        else        canvas_draw_circle(canvas, cx, 58, 3);
    }
}

// falling phase - bm playback + thin top progress bar.
// 382 src frames squashed into the 2.5s window: most get skipped at 30fps redraw
// but that's the chosen "speed up" pacing.
static void drop_view_draw_falling(Canvas* canvas, DropModel* m) {
    uint32_t local = m->now_frame - m->phase_start_frame;
    if(local > FALLING_FRAMES) local = FALLING_FRAMES;

    const uint8_t fc = A_drop_animation_128x64.frame_count;
    uint32_t src_idx = (local * fc) / FALLING_FRAMES;
    if(src_idx >= fc) src_idx = fc - 1;
    canvas_draw_bitmap(canvas, 0, 0, 128, 64, A_drop_animation_128x64.frames[src_idx]);

    // top progress bar - frame outline + 1px fill, grows L->R over 2.5s
    uint16_t fill = (local * 126) / FALLING_FRAMES;
    canvas_draw_frame(canvas, 0, 2, 128, 3);
    if(fill > 0) canvas_draw_box(canvas, 1, 3, fill > 126 ? 126 : fill, 1);
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

static void drop_view_draw(Canvas* canvas, void* model) {
    DropModel* m = model;
    canvas_clear(canvas);
    canvas_set_color(canvas, ColorBlack);

    switch(m->phase) {
    case DropPhaseArming:    drop_view_draw_arming(canvas);       break;
    case DropPhaseCountdown: drop_view_draw_countdown(canvas, m); break;
    case DropPhaseFalling:   drop_view_draw_falling(canvas, m);   break;
    case DropPhaseDone:      drop_view_draw_done(canvas);         break;
    }
}

static bool drop_view_input(InputEvent* event, void* context) {
    View* view = context;
    bool consumed = false;

    if(event->type != InputTypeShort && event->type != InputTypePress) return false;

    with_view_model(
        view,
        DropModel * m,
        {
            switch(m->phase) {
            case DropPhaseArming:
                if(event->key == InputKeyOk) {
                    m->phase = DropPhaseCountdown;
                    m->phase_start_frame = m->now_frame;
                    m->cue_played = false;
                    consumed = true;
                }
                break;
            case DropPhaseCountdown: break;     // wait for tick
            case DropPhaseFalling:   break;     // wait for tick
            case DropPhaseDone:
                if(event->key == InputKeyOk) {
                    m->phase = DropPhaseArming;
                    consumed = true;
                }
                break;
            }
        },
        true);

    return consumed;
}

static void drop_view_enter(void* context) {
    View* view = context;
    with_view_model(
        view,
        DropModel * m,
        {
            m->phase = DropPhaseArming;
            m->now_frame = 0;
            m->phase_start_frame = 0;
        },
        true);
}

// back-out mid-fall: session still open, has to close before the view goes
static void drop_view_exit(void* context) {
    View* view = context;
    with_view_model(
        view,
        DropModel * m,
        {
            if(m->phase == DropPhaseFalling) {
                // DATA: collector stop (aborted - row count < FALLING_FRAMES)
            }
        },
        false);
}

void drop_view_tick(View* view, uint32_t frame) {
    with_view_model(
        view,
        DropModel * m,
        {
            m->now_frame = frame;
            uint32_t local = frame - m->phase_start_frame;

            // fire the "DROP!" cue at the start of the 4th countdown second
            if(m->phase == DropPhaseCountdown && !m->cue_played && local >= 3 * 30) {
                m->cue_played = true;
                if(drop_notif) notification_message(drop_notif, &sequence_audiovisual_alert);
            }

            if(m->phase == DropPhaseCountdown && local >= COUNTDOWN_FRAMES) {
                m->phase = DropPhaseFalling;
                m->phase_start_frame = frame;
                // DATA: collector start (2.5s window opens)
            } else if(m->phase == DropPhaseFalling && local >= FALLING_FRAMES) {
                m->phase = DropPhaseDone;
                m->phase_start_frame = frame;
                // DATA: collector stop
            }
        },
        true);
}

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
