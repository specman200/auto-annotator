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
