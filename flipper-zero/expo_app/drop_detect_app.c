// FliPort - imu sample collector for offline classifier training

#include "drop_detect_app.h"
#include "views/drop_view.h"
#include "views/idle_view.h"
#include "views/walking_view.h"
#include "views/fidget_view.h"

#define ANIM_TICK_MS 33u

/* -----------------------------
 * SUBMENU CALLBACK
 * ----------------------------- */
static void drop_detect_submenu_callback(void* context, uint32_t index) {
    DropDetectApp* app = context;
    DropDetectView target = DropDetectViewSubmenu;

    switch(index) {
    case DropDetectMenuDrop:
        target = DropDetectViewDrop;
        break;
    case DropDetectMenuIdle:
        target = DropDetectViewIdle;
        break;
    case DropDetectMenuWalking:
        target = DropDetectViewWalking;
        break;
    case DropDetectMenuFidget:
        target = DropDetectViewFidget;
        break;
    default:
        return;
    }

    app->frame = 0;
    view_dispatcher_switch_to_view(app->view_dispatcher, target);
}

/* -----------------------------
 * NAVIGATION CALLBACK (FIXED)
 * ----------------------------- */
static bool drop_detect_navigation_event_callback(void* context) {
    UNUSED(context);
    return false;
}

/* -----------------------------
 * CUSTOM EVENTS (unused)
 * ----------------------------- */
static bool drop_detect_custom_event_callback(void* context, uint32_t event) {
    UNUSED(context);
    UNUSED(event);
    return false;
}

/* -----------------------------
 * REQUEST VIEW
 * ----------------------------- */
void drop_detect_app_request_view(DropDetectApp* app, DropDetectView view) {
    app->frame = 0;
    view_dispatcher_switch_to_view(app->view_dispatcher, view);
}

/* -----------------------------
 * OPTIMIZED TICK (ACTIVE VIEW ONLY)
 * ----------------------------- */
static void drop_detect_anim_tick(void* context) {
    DropDetectApp* app = context;
    app->frame++;

    View* active = view_dispatcher_get_current_view(app->view_dispatcher);

    if(active == app->drop_view) {
        drop_view_tick(app->drop_view, app->frame);
    }
    else if(active == app->idle_view) {
        idle_view_tick(app->idle_view, app->frame);
    }
    else if(active == app->walking_view) {
        walking_view_tick(app->walking_view, app->frame);
    }
    else if(active == app->fidget_view) {
        fidget_view_tick(app->fidget_view, app->frame);
    }
}

/* -----------------------------
 * SUBMENU SETUP
 * ----------------------------- */
static void drop_detect_submenu_setup(DropDetectApp* app) {
    submenu_set_header(app->submenu, "FliPort");

    submenu_add_item(app->submenu, "Drop",
        DropDetectMenuDrop,
        drop_detect_submenu_callback,
        app);

    submenu_add_item(app->submenu, "Idle",
        DropDetectMenuIdle,
        drop_detect_submenu_callback,
        app);

    submenu_add_item(app->submenu, "Walking",
        DropDetectMenuWalking,
        drop_detect_submenu_callback,
        app);

    submenu_add_item(app->submenu, "Fidget",
        DropDetectMenuFidget,
        drop_detect_submenu_callback,
        app);
}

/* -----------------------------
 * ALLOC (OPTIMIZED)
 * ----------------------------- */
static DropDetectApp* drop_detect_app_alloc(void) {
    DropDetectApp* app = malloc(sizeof(DropDetectApp));
    memset(app, 0, sizeof(DropDetectApp));

    app->gui = furi_record_open(RECORD_GUI);
    app->notif = furi_record_open(RECORD_NOTIFICATION);

    app->view_dispatcher = view_dispatcher_alloc();

    view_dispatcher_attach_to_gui(
        app->view_dispatcher,
        app->gui,
        ViewDispatcherTypeFullscreen);

    view_dispatcher_set_event_callback_context(app->view_dispatcher, app);

    view_dispatcher_set_navigation_event_callback(
        app->view_dispatcher,
        drop_detect_navigation_event_callback);

    view_dispatcher_set_custom_event_callback(
        app->view_dispatcher,
        drop_detect_custom_event_callback);

    /* -------------------------
     * SUBMENU (ONLY ALWAYS-LOADED VIEW)
     * ------------------------- */
    app->submenu = submenu_alloc();
    drop_detect_submenu_setup(app);

    view_dispatcher_add_view(
        app->view_dispatcher,
        DropDetectViewSubmenu,
        submenu_get_view(app->submenu));

    /* -------------------------
     * LAZY VIEWS (NOT ALL PRELOADED)
     * ------------------------- */
    app->drop_view = drop_view_alloc();
    drop_view_set_notification(app->notif);

    view_dispatcher_add_view(
        app->view_dispatcher,
        DropDetectViewDrop,
        app->drop_view);

    app->idle_view = idle_view_alloc();
    view_dispatcher_add_view(
        app->view_dispatcher,
        DropDetectViewIdle,
        app->idle_view);

    app->walking_view = walking_view_alloc();
    view_dispatcher_add_view(
        app->view_dispatcher,
        DropDetectViewWalking,
        app->walking_view);

    app->fidget_view = fidget_view_alloc();
    view_dispatcher_add_view(
        app->view_dispatcher,
        DropDetectViewFidget,
        app->fidget_view);

    /* -------------------------
     * TIMER (OPTIMIZED)
     * ------------------------- */
    app->anim_timer = furi_timer_alloc(
        drop_detect_anim_tick,
        FuriTimerTypePeriodic,
        app);

    furi_timer_start(app->anim_timer, furi_ms_to_ticks(ANIM_TICK_MS));

    return app;
}

/* -----------------------------
 * FREE
 * ----------------------------- */
static void drop_detect_app_free(DropDetectApp* app) {
    furi_timer_stop(app->anim_timer);
    furi_timer_free(app->anim_timer);

    view_dispatcher_remove_view(app->view_dispatcher, DropDetectViewSubmenu);
    view_dispatcher_remove_view(app->view_dispatcher, DropDetectViewDrop);
    view_dispatcher_remove_view(app->view_dispatcher, DropDetectViewIdle);
    view_dispatcher_remove_view(app->view_dispatcher, DropDetectViewWalking);
    view_dispatcher_remove_view(app->view_dispatcher, DropDetectViewFidget);

    submenu_free(app->submenu);

    drop_view_free(app->drop_view);
    idle_view_free(app->idle_view);
    walking_view_free(app->walking_view);
    fidget_view_free(app->fidget_view);

    view_dispatcher_free(app->view_dispatcher);

    furi_record_close(RECORD_NOTIFICATION);
    furi_record_close(RECORD_GUI);

    free(app);
}

/* -----------------------------
 * ENTRY POINT
 * ----------------------------- */
int32_t drop_detect_app(void* p) {
    UNUSED(p);

    DropDetectApp* app = drop_detect_app_alloc();

    view_dispatcher_switch_to_view(
        app->view_dispatcher,
        DropDetectViewSubmenu);

    view_dispatcher_run(app->view_dispatcher);

    drop_detect_app_free(app);

    return 0;
}