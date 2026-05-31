/*
 * drop_dect_svc.c — headless background-service entry point.
 *
 * This is the "drop-in for custom firmware" variant: no GUI, no view dispatcher.
 * It loads the model once at boot and then runs the shared monitor loop forever,
 * sounding the siren whenever a drop is detected — even with no app open.
 *
 * Because stock firmware unloads FAPs on exit, a true always-on service must be
 * compiled into a firmware image (apptype=FlipperAppType.SERVICE). See README.md
 * for the manifest block and integration steps. The detection logic itself is
 * identical to the FAP — both call monitor_run().
 */
#include <furi.h>
#include <storage/storage.h>

#include <stdlib.h>

#include "model.h"
#include "monitor.h"

#define TAG "drop_dect_svc"

/* Where the service expects its weights. When integrating into firmware, deploy
 * model.json here (apps_assets path for the service appid). */
#define SVC_MODEL_PATH "/ext/apps_assets/drop_detect_svc/model.json"

#define MODEL_LOAD_ATTEMPTS 10
#define MODEL_LOAD_RETRY_MS 1000u
#define DROP_THRESHOLD 0.5f

int32_t drop_dect_svc(void* p) {
    UNUSED(p);

    DropModel* model = malloc(sizeof(DropModel));
    if(!model) return -1;

    /* SD may mount slightly after services start; retry a few times rather than
     * crashing the boot sequence if the model isn't readable yet. */
    bool loaded = false;
    for(int attempt = 0; attempt < MODEL_LOAD_ATTEMPTS && !loaded; attempt++) {
        loaded = model_load_json(model, SVC_MODEL_PATH);
        if(!loaded) {
            FURI_LOG_W(TAG, "model load attempt %d/%d failed", attempt + 1, MODEL_LOAD_ATTEMPTS);
            furi_delay_ms(MODEL_LOAD_RETRY_MS);
        }
    }

    if(!loaded) {
        FURI_LOG_E(TAG, "no model at %s; service idle", SVC_MODEL_PATH);
        free(model);
        return 0; /* degrade gracefully — do not block the system */
    }

    MonitorCfg cfg = {0};
    cfg.model = model;
    cfg.running = true; /* headless service: run indefinitely */
    cfg.thresh = DROP_THRESHOLD;
    cfg.on_verdict = NULL; /* monitor.c already logs the verdict */
    cfg.ud = NULL;
    cfg.notif = NULL; /* no notification record held; speaker-only alarm */

    FURI_LOG_I(TAG, "drop service running");
    monitor_run(&cfg); /* normally never returns (cfg.running stays true) */

    free(model);
    return 0;
}
