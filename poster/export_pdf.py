import sys, os
from playwright.sync_api import sync_playwright

html_path = os.path.join(os.path.dirname(__file__), "poster.html")
out_path = os.path.join(os.path.dirname(__file__), "..", "files", "SCIENCEMONTGOMERY_POSTER.pdf")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(
        viewport={"width": int(48 * 96), "height": int(36 * 96)},
        device_scale_factor=2,
    )
    page.goto(f"file://{html_path}", wait_until="networkidle")
    page.wait_for_timeout(2000)
    page.pdf(
        path=out_path,
        width="48in",
        height="36in",
        print_background=True,
        scale=1.0,
    )
    browser.close()
    sz = os.path.getsize(out_path)
    print(f"PDF: {out_path}  ({sz/1024:.0f} KB)")
