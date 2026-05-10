# FliPort

flipper zero imu sample collector for offline classifier training.
four screens, one per action label: drop / idle / walking / fidget.

## install on a flipper

prereqs: python 3.10+, a flipper zero on usb, qFlipper closed.

one-time setup:

```powershell
python -m pip install --upgrade ufbt
ufbt update
```

build + flash from this folder:

```powershell
ufbt launch
```

builds, copies `dist/drop_detect.fap` to the device, and starts it.
on the flipper: `Apps -> Tools -> FliPort`.

if `ufbt launch` can't see the device, just `ufbt` to build, then drop
`dist/drop_detect.fap` into `apps/Tools/` on the sd card via qFlipper.

## data collector - where to plug in

ui is done. imu + persistence is the next piece. anchors:

- `drop_detect_app.c`
  - in `drop_detect_app_alloc`: open the lsm6dso, alloc a 50hz furi_timer
    (separate from the 33ms anim tick), leave the timer stopped.
    the comment `// imu + sampler timer hook up here.` marks the spot.
  - in `drop_detect_app_free`: close the driver, free the timer.

- `views/drop_view.c`
  drop is event-shaped, not continuous. capture is gated to the 2.5s
  falling phase only. anchors live inside `drop_view_tick`:
    `// DATA: collector start (2.5s window opens)`
    `// DATA: collector stop`
  plus `drop_view_exit` handles mid-fall back-out (session still open).

- `views/idle_view.c`, `walking_view.c`, `fidget_view.c`
  capture = whole view lifetime. each has a matching pair:
    `// DATA: collector start`   in `<name>_view_enter`
    `// DATA: collector stop`    in `<name>_view_exit`

each session writes one labeled csv on the sd card; pick the path/format
when wiring it up.
