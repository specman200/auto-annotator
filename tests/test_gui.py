"""Browser tests for the annotation GUI.

Layout bugs are invisible to the other tests: a panel that overflows its box
swallows clicks meant for the rows underneath, and everything still "works"
as far as the API is concerned. These drive a real browser instead.

Skipped unless playwright and a chromium build are both installed::

    pip install playwright && playwright install chromium
"""

import glob
import os
import socket
import threading
import time

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

from auto_annotator.server import create_app  # noqa: E402


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def chromium_candidates():
    """Places a usable chromium may live, best first.

    ``launch()`` only finds the exact build the installed playwright expects,
    so fall back to any build present before giving up — CI images often ship
    one that is a version or two off.
    """
    yield os.environ.get("AUTO_ANNOTATOR_CHROMIUM")
    yield None  # playwright's own registry
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if root:
        for pattern in (
            "chromium-*/chrome-linux/chrome",
            "chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
            "chromium-*/chrome-win/chrome.exe",
            "chromium_headless_shell-*/chrome-linux/headless_shell",
        ):
            yield from sorted(glob.glob(os.path.join(root, pattern)), reverse=True)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        instance = None
        for candidate in chromium_candidates():
            if candidate is not None and not os.path.exists(candidate):
                continue
            try:
                instance = pw.chromium.launch(executable_path=candidate)
                break
            except Exception:
                continue
        if instance is None:
            pytest.skip("no usable chromium for playwright")
        yield instance
        instance.close()


@pytest.fixture
def gui(browser, tmp_path_factory):
    """A served project with enough boxes to overflow the sidebar."""
    import uvicorn
    from PIL import Image

    images = tmp_path_factory.mktemp("gui-images")
    for index in range(3):
        Image.new("RGB", (640, 480), "#203040").save(images / f"img_{index}.png")

    app = create_app(images, model_spec="mock", conf=0.2, classes=["cat", "dog"])
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    else:
        pytest.skip("server did not start")

    page = browser.new_page(viewport={"width": 1280, "height": 800})
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
    page.wait_for_timeout(400)

    # 16 boxes is more than the sidebar can show at once.
    annotations = [
        {
            "label": "cat" if index % 2 else "dog",
            "box": {
                "x1": 0.02 + (index % 4) * 0.2, "y1": 0.02 + (index // 4) * 0.2,
                "x2": 0.14 + (index % 4) * 0.2, "y2": 0.14 + (index // 4) * 0.2,
            },
            "score": 0.5, "source": "model",
        }
        for index in range(16)
    ]
    page.evaluate(
        """async (annotations) => {
            await fetch('/api/images/img_0.png/annotations', {
                method: 'PUT', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({annotations, status: 'predicted'})});
        }""",
        annotations,
    )
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(500)

    yield page, errors
    page.close()
    server.should_exit = True


def test_a_long_box_list_never_covers_the_panel_below(gui):
    """The regression that made predicted boxes uneditable."""
    page, _ = gui
    geometry = page.evaluate(
        """() => {
            const rect = (sel) => {
                const box = document.querySelector(sel).getBoundingClientRect();
                return {top: box.top, bottom: box.bottom};
            };
            return {
                scroller: rect('#panel-boxes .panel-body'),
                boxes: rect('#panel-boxes'),
                shortcuts: rect('#panel-shortcuts'),
            };
        }"""
    )
    assert geometry["scroller"]["bottom"] <= geometry["boxes"]["bottom"] + 1
    assert geometry["boxes"]["bottom"] <= geometry["shortcuts"]["top"] + 1


def test_every_visible_box_row_can_be_clicked(gui):
    """A row hidden behind another panel cannot be selected, relabelled or deleted."""
    page, _ = gui
    blocked = page.evaluate(
        """() => {
            const host = document.querySelector('#panel-boxes .panel-body')
                .getBoundingClientRect();
            const bad = [];
            document.querySelectorAll('#annotation-list li').forEach((row, index) => {
                const box = row.getBoundingClientRect();
                if (box.top < host.top || box.bottom > host.bottom) return;
                const hit = document.elementFromPoint(box.left + 8, box.top + box.height / 2);
                if (!(hit && (hit === row || row.contains(hit)))) bad.push(index);
            });
            return bad;
        }"""
    )
    assert blocked == []
    assert page.evaluate(
        "document.querySelector('#panel-boxes .panel-body').scrollHeight > "
        "document.querySelector('#panel-boxes .panel-body').clientHeight"
    ), "the list should scroll rather than grow past its panel"


def test_a_predicted_box_can_be_selected_and_edited(gui):
    """Selecting from the list and nudging is the workflow for dense predictions."""
    page, _ = gui
    box_id = page.evaluate("state.record.annotations[5].id")
    page.locator(f'#annotation-list li[data-id="{box_id}"] .src').click()
    page.wait_for_timeout(200)
    assert page.evaluate("state.selectedId") == box_id

    read = "(id) => state.record.annotations.find(a => a.id === id).box"
    before = page.evaluate(read, box_id)
    for _ in range(5):
        page.keyboard.press("ArrowRight")
    page.wait_for_timeout(300)
    moved = page.evaluate(read, box_id)
    assert moved["x1"] > before["x1"]

    for _ in range(5):
        page.keyboard.press("Shift+ArrowDown")
    page.wait_for_timeout(300)
    assert page.evaluate(read, box_id)["y2"] > moved["y2"]
    assert page.evaluate(
        "(id) => state.record.annotations.find(a => a.id === id).source", box_id
    ) == "human"


def test_the_cursor_says_what_a_drag_will_do(gui):
    """Handles are only discoverable if the cursor changes over them."""
    page, _ = gui
    box_id = page.evaluate("state.record.annotations[0].id")
    page.evaluate("(id) => selectAnnotation(id)", box_id)
    spots = page.evaluate(
        """(id) => {
            const a = state.record.annotations.find(x => x.id === id);
            const r = canvas.getBoundingClientRect();
            const at = (nx, ny) => {
                const p = imageToScreen(nx * img.naturalWidth, ny * img.naturalHeight);
                return {x: r.left + p.x, y: r.top + p.y};
            };
            return {
                corner: at(a.box.x2, a.box.y2),
                inside: at((a.box.x1 + a.box.x2) / 2, (a.box.y1 + a.box.y2) / 2),
                empty: at(0.97, 0.97),
            };
        }""",
        box_id,
    )
    for spot, expected in (("corner", "resize"), ("inside", "move"), ("empty", "crosshair")):
        page.mouse.move(spots[spot]["x"], spots[spot]["y"])
        page.wait_for_timeout(150)
        assert expected in page.evaluate("canvas.style.cursor"), f"over the {spot}"


def test_panels_resize_and_stay_that_way(gui):
    page, _ = gui
    height = "document.getElementById('panel-classes').getBoundingClientRect().height"
    before = page.evaluate(height)
    splitter = page.locator(".splitter.horizontal").first.bounding_box()
    page.mouse.move(splitter["x"] + splitter["width"] / 2, splitter["y"] + 2)
    page.mouse.down()
    page.mouse.move(splitter["x"] + splitter["width"] / 2, splitter["y"] + 90, steps=8)
    page.mouse.up()
    page.wait_for_timeout(300)
    assert page.evaluate(height) > before + 40

    page.locator("#panel-shortcuts .panel-head").click()
    page.wait_for_timeout(200)
    collapsed = "document.getElementById('panel-shortcuts').classList.contains('collapsed')"
    assert page.evaluate(collapsed)

    page.reload(wait_until="networkidle")
    page.wait_for_timeout(500)
    assert page.evaluate(collapsed), "collapse state should survive a reload"
    assert page.evaluate(height) > before + 40, "panel size should survive a reload"


def test_the_gui_runs_without_javascript_errors(gui):
    page, errors = gui
    page.click("#btn-predict")
    page.wait_for_timeout(700)
    page.keyboard.press("n")
    page.wait_for_timeout(500)
    assert errors == []


def test_a_stored_layout_can_never_squeeze_the_image_away(gui):
    """A few enthusiastic drags used to persist a layout that wrecked the UI."""
    page, _ = gui
    page.evaluate(
        """() => localStorage.setItem('auto-annotator.layout', JSON.stringify({
            leftWidth: 620, rightWidth: 620,
            panels: {'panel-classes': 690}, collapsed: []}))"""
    )
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(600)

    geometry = page.evaluate(
        """() => {
            const rect = (sel) => {
                const box = document.querySelector(sel).getBoundingClientRect();
                return {width: box.width, height: box.height, bottom: box.bottom};
            };
            return {canvas: rect('#canvas'), boxes: rect('#panel-boxes'),
                    shortcuts: rect('#panel-shortcuts'), height: window.innerHeight};
        }"""
    )
    assert geometry["canvas"]["width"] >= 300, "the image keeps a usable share"
    assert geometry["boxes"]["height"] >= 110, "the box list stays usable"
    assert geometry["shortcuts"]["bottom"] <= geometry["height"] + 1, "nothing off-screen"


def test_shrinking_the_window_reflows_instead_of_crushing_the_canvas(gui):
    page, _ = gui
    page.set_viewport_size({"width": 820, "height": 620})
    page.wait_for_timeout(500)
    width = page.evaluate("document.getElementById('canvas').getBoundingClientRect().width")
    assert width >= 200
    page.set_viewport_size({"width": 1280, "height": 800})
    page.wait_for_timeout(400)


def test_reset_layout_recovers_a_wedged_sidebar(gui):
    page, _ = gui
    page.evaluate(
        """() => localStorage.setItem('auto-annotator.layout', JSON.stringify({
            leftWidth: 600, rightWidth: 600,
            panels: {'panel-classes': 600}, collapsed: ['panel-boxes']}))"""
    )
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(500)
    page.click("#btn-reset-layout")
    page.wait_for_timeout(400)

    assert page.evaluate("document.querySelectorAll('.panel.collapsed').length") == 0
    assert page.evaluate("document.getElementById('canvas').getBoundingClientRect().width") > 500
    assert page.evaluate("localStorage.getItem('auto-annotator.layout')") in (None, "null")


def test_a_box_hidden_under_another_can_still_be_selected(gui):
    """Clicking the same spot cycles the stack, innermost first."""
    page, _ = gui
    page.evaluate(
        """async () => {
            const annotations = [
              {label: 'cat', box: {x1: .1, y1: .1, x2: .9, y2: .9}, score: .9, source: 'model'},
              {label: 'dog', box: {x1: .3, y1: .3, x2: .6, y2: .6}, score: .8, source: 'model'},
              {label: 'cat', box: {x1: .4, y1: .4, x2: .5, y2: .5}, score: .7, source: 'model'}];
            await fetch('/api/images/img_0.png/annotations', {
                method: 'PUT', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({annotations, status: 'predicted'})});
        }"""
    )
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(600)

    spot = page.evaluate(
        """() => { const r = canvas.getBoundingClientRect();
            const p = imageToScreen(0.45 * img.naturalWidth, 0.45 * img.naturalHeight);
            return {x: r.left + p.x, y: r.top + p.y}; }"""
    )
    seen = []
    for _ in range(3):
        page.mouse.click(spot["x"], spot["y"])
        page.wait_for_timeout(250)
        seen.append(page.evaluate("state.selectedId"))
    assert len(set(seen)) == 3, "each click should reach a different box in the stack"


def test_the_gui_is_served_with_revalidation(gui):
    """Mismatched cached HTML/CSS renders a broken layout that looks like a bug."""
    page, _ = gui
    headers = {}
    page.on(
        "response",
        lambda response: headers.__setitem__(
            response.url.rsplit("/", 1)[-1] or "index",
            response.headers.get("cache-control", ""),
        ),
    )
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(300)
    for name in ("app.js", "style.css"):
        assert "no-cache" in headers.get(name, ""), f"{name}: {headers.get(name)!r}"


@pytest.fixture
def one_big_box(gui):
    """An image holding a single box that covers most of it."""
    page, errors = gui
    page.evaluate(
        """async () => {
            await fetch('/api/images/img_1.png/annotations', {
                method: 'PUT', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({annotations: [{label: 'cat',
                    box: {x1: .1, y1: .1, x2: .9, y2: .9}, score: .9, source: 'model'}],
                    status: 'predicted'})});
        }"""
    )
    page.evaluate("() => openImage('img_1.png')")
    page.wait_for_timeout(600)
    return page, errors


def canvas_point(page, nx, ny):
    return page.evaluate(
        """([nx, ny]) => { const r = canvas.getBoundingClientRect();
            const p = imageToScreen(nx * img.naturalWidth, ny * img.naturalHeight);
            return {x: r.left + p.x, y: r.top + p.y}; }""",
        [nx, ny],
    )


def test_shift_drag_draws_a_box_inside_another(one_big_box):
    """Nested labels (a wheel inside a car) need a drag that beats the move gesture."""
    page, _ = one_big_box
    parent = page.evaluate("state.record.annotations[0].box")
    start, end = canvas_point(page, 0.3, 0.3), canvas_point(page, 0.5, 0.5)

    page.keyboard.down("Shift")
    page.mouse.move(start["x"], start["y"])
    page.mouse.down()
    page.mouse.move(end["x"], end["y"], steps=10)
    page.mouse.up()
    page.keyboard.up("Shift")
    page.wait_for_timeout(500)

    boxes = page.evaluate("state.record.annotations")
    assert len(boxes) == 2, "shift+drag should add a box, not move the parent"
    assert boxes[0]["box"] == parent, "the parent must not move"
    child = boxes[1]["box"]
    assert child["x1"] >= parent["x1"] and child["x2"] <= parent["x2"]
    assert child["y1"] >= parent["y1"] and child["y2"] <= parent["y2"]
    assert page.evaluate("state.selectedId") == boxes[1]["id"]


def test_a_plain_drag_inside_a_box_still_moves_it(one_big_box):
    page, _ = one_big_box
    before = page.evaluate("state.record.annotations[0].box")
    start, end = canvas_point(page, 0.3, 0.3), canvas_point(page, 0.4, 0.4)
    page.mouse.move(start["x"], start["y"])
    page.mouse.down()
    page.mouse.move(end["x"], end["y"], steps=8)
    page.mouse.up()
    page.wait_for_timeout(500)
    assert page.evaluate("state.record.annotations.length") == 1
    assert page.evaluate("state.record.annotations[0].box.x1") > before["x1"]


def test_draw_mode_nests_without_holding_shift(one_big_box):
    page, _ = one_big_box
    page.keyboard.press("d")
    page.wait_for_timeout(200)
    assert page.evaluate("state.drawMode") is True

    start, end = canvas_point(page, 0.6, 0.6), canvas_point(page, 0.75, 0.78)
    page.mouse.move(start["x"], start["y"])
    page.mouse.down()
    page.mouse.move(end["x"], end["y"], steps=10)
    page.mouse.up()
    page.wait_for_timeout(500)
    assert page.evaluate("state.record.annotations.length") == 2

    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    assert page.evaluate("state.drawMode") is False


def test_an_edit_during_a_save_is_not_lost(one_big_box):
    """A slow save's reply must not overwrite boxes edited while it was in flight."""
    page, _ = one_big_box
    page.evaluate(
        """() => {
            const original = window.fetch;
            window.fetch = async (...args) => {
                const response = await original(...args);
                if (String(args[0]).includes('/annotations')) {
                    await new Promise((resolve) => setTimeout(resolve, 900));
                }
                return response;
            };
        }"""
    )
    page.evaluate(
        """() => { const a = state.record.annotations[0];
            a.box = {...a.box, x1: a.box.x1 + 0.2, x2: a.box.x2 + 0.2}; markDirty(); }"""
    )
    page.wait_for_timeout(600)
    page.evaluate(
        """() => { const a = state.record.annotations[0];
            a.box = {...a.box, y1: a.box.y1 + 0.3, y2: a.box.y2 + 0.3}; markDirty(); }"""
    )
    page.wait_for_timeout(3000)

    box = page.evaluate("state.record.annotations[0].box")
    assert box["x1"] == pytest.approx(0.3, abs=1e-6), "first edit lost"
    assert box["y1"] == pytest.approx(0.4, abs=1e-6), "second edit lost to a stale reply"
