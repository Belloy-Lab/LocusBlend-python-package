"""Streamlit-independent image and export helpers.

Extracted from ``app.2.8.12.py``: Plotly-to-PNG rendering (kaleido), Pillow
loading helpers, figure cloning (re-exported from :mod:`locusblend.plotting`),
generic image composition utilities, and the publication-page canvas builder.

Deliberately NOT included: Streamlit download buttons, Streamlit session-state
access, and ``make_locusblend_export_bytes`` (which reads the cached figures
from session state). That wrapper stays in the app; the reusable part of it is
``image_to_export_bytes`` below.

App coupling removed: ``draw_export_header`` / ``build_locusblend_export_image``
used to read the legend images from the app's ``assets/`` directory next to the
script. The directory is now an explicit ``asset_dir`` parameter (default
``None`` = no legend images).
"""

from __future__ import annotations

import base64
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path

import plotly.io as pio

from .plotting import clone_plotly_figure

__all__ = [
    "build_locusblend_export_image",
    "clone_plotly_figure",
    "draw_export_header",
    "image_to_data_uri",
    "image_to_export_bytes",
    "open_pil_asset",
    "open_pil_image_from_bytes",
    "paste_contained",
    "render_plotly_figure_to_png_bytes",
    "resize_contained",
]

# Legend images used by the publication-page header (baseline asset names).
LEGEND_ASSET_NAMES = ("legend_overlay.png", "Drawing3.png")

# --- extracted from app.2.8.12.py (verbatim) ---


def image_to_data_uri(path):
    mime = guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise RuntimeError(
            "Pillow is required to compose PNG/PDF exports. Install it with: pip install pillow"
        ) from e
    return Image, ImageDraw, ImageFont


def _get_resample_filter():
    Image, _, _ = _load_pillow()
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    if hasattr(Image, "LANCZOS"):
        return Image.LANCZOS
    return Image.BICUBIC


def _pil_font(size, bold=False):
    _, _, ImageFont = _load_pillow()
    candidates = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, int(size))
        except Exception:
            pass
    return ImageFont.load_default()


def _text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), str(text), font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            return draw.textsize(str(text), font=font)
        except Exception:
            return len(str(text)) * 7, 12


def _draw_wrapped_text(draw, text, xy, max_width, font, fill, line_spacing=4):
    x, y = xy
    max_width = max(1, int(max_width))

    def draw_line(line_text, line_y):
        draw.text((x, line_y), line_text, font=font, fill=fill)
        _, line_h = _text_size(draw, line_text or "Ag", font)
        return line_y + line_h + line_spacing

    for raw_line in str(text or "").splitlines():
        words = raw_line.split()
        if not words:
            y = draw_line("", y)
            continue

        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            candidate_w, _ = _text_size(draw, candidate, font)
            if candidate_w <= max_width or not line:
                line = candidate
            else:
                y = draw_line(line, y)
                line = word
        if line:
            y = draw_line(line, y)
    return y


def _format_export_context(context):
    context = dict(context or {})
    lines = []

    mode = context.get("mode")
    compare_mode = context.get("compare_mode")
    if mode:
        lines.append(f"Mode: {mode}")
    if compare_mode:
        lines.append(f"Compare: {compare_mode}")

    chrom = context.get("chromosome")
    center_bp = context.get("center_bp")
    window_kb = context.get("window_kb")
    locus_parts = []
    if chrom not in (None, ""):
        locus_parts.append(f"chr{chrom}")
    if center_bp not in (None, ""):
        try:
            locus_parts.append(f"center {int(center_bp):,} bp")
        except Exception:
            locus_parts.append(f"center {center_bp} bp")
    if window_kb not in (None, ""):
        try:
            locus_parts.append(f"+/- {int(window_kb):,} kb")
        except Exception:
            locus_parts.append(f"+/- {window_kb} kb")
    if locus_parts:
        lines.append("Locus: " + ", ".join(locus_parts))

    title_top = context.get("title_top")
    title_bottom = context.get("title_bottom")
    title_parts = [str(v) for v in [title_top, title_bottom] if v not in (None, "")]
    if title_parts:
        lines.append("Sources: " + " / ".join(title_parts))

    highlight_genes = str(context.get("highlight_genes", "") or "").strip()
    if highlight_genes:
        lines.append(f"Highlighted genes: {highlight_genes}")

    ld_status_caption = str(context.get("ld_status_caption", "") or "").strip()
    if ld_status_caption:
        lines.append(ld_status_caption)

    return "\n".join(lines)


def render_plotly_figure_to_png_bytes(fig, width_px, height_px, scale=1):
    """Render a cloned Plotly figure to PNG bytes using kaleido.

    This must not mutate the cached session_state figure.
    """
    if fig is None:
        raise ValueError("No Plotly figure is available for export.")

    width_px = max(1, int(width_px))
    height_px = max(1, int(height_px))

    fig_copy = clone_plotly_figure(fig)
    fig_copy.update_layout(
        width=width_px,
        height=height_px,
        autosize=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )

    try:
        return pio.to_image(
            fig_copy,
            format="png",
            width=width_px,
            height=height_px,
            scale=scale,
        )
    except Exception as e:
        raise RuntimeError(
            "Static export failed. Plotly image export requires kaleido. "
            "Depending on the installed kaleido version, Chrome may also be required on the server."
        ) from e


def open_pil_image_from_bytes(data):
    Image, _, _ = _load_pillow()
    return Image.open(BytesIO(data)).convert("RGBA")


def open_pil_asset(path):
    if path is None:
        return None
    try:
        path = Path(path)
        if not path.exists():
            return None
        Image, _, _ = _load_pillow()
        return Image.open(path).convert("RGBA")
    except Exception:
        return None


def resize_contained(img, max_w, max_h):
    if img is None:
        return None
    max_w = max(1, int(max_w))
    max_h = max(1, int(max_h))
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return img.resize((1, 1), _get_resample_filter())
    scale = min(max_w / src_w, max_h / src_h)
    target_w = max(1, int(round(src_w * scale)))
    target_h = max(1, int(round(src_h * scale)))
    return img.resize((target_w, target_h), _get_resample_filter())


def paste_contained(canvas, img, box, align="center"):
    if canvas is None or img is None:
        return None

    x, y, w, h = [int(v) for v in box]
    fitted = resize_contained(img, w, h)
    if fitted is None:
        return None

    if align == "left":
        paste_x = x
    elif align == "right":
        paste_x = x + max(0, w - fitted.width)
    else:
        paste_x = x + max(0, (w - fitted.width) // 2)
    paste_y = y + max(0, (h - fitted.height) // 2)

    if fitted.mode == "RGBA":
        canvas.paste(fitted, (paste_x, paste_y), fitted)
    else:
        canvas.paste(fitted, (paste_x, paste_y))
    return paste_x, paste_y, fitted.width, fitted.height


def draw_export_header(canvas, draw, context, include_legends, header_box, asset_dir=None):
    """Draw the publication-page header (title, context, optional legend images).

    app.2.8.12.py read the legend images from the app's ``assets/`` directory.
    Here ``asset_dir`` is explicit; ``None`` simply means no legend images.
    """
    x, y, w, h = [int(v) for v in header_box]
    pad = max(12, int(h * 0.10))
    border = "#d1d5db"
    fill = "#f8fafc"
    text = "#111827"
    muted = "#374151"

    try:
        draw.rounded_rectangle([x, y, x + w, y + h], radius=max(8, h // 18), fill=fill, outline=border, width=1)
    except Exception:
        draw.rectangle([x, y, x + w, y + h], fill=fill, outline=border, width=1)

    title_font = _pil_font(max(18, min(44, h // 7)), bold=True)
    body_font = _pil_font(max(11, min(24, h // 15)), bold=False)

    legend_imgs = []
    if include_legends:
        for asset_name in LEGEND_ASSET_NAMES:
            img = open_pil_asset(None if asset_dir is None else Path(asset_dir) / asset_name)
            if img is not None:
                legend_imgs.append(img)

    inner_w = max(1, w - 2 * pad)
    legend_w = int(inner_w * 0.36) if legend_imgs else 0
    legend_gap = max(8, pad // 2)
    text_w = inner_w - legend_w - (legend_gap if legend_imgs else 0)
    text_x = x + pad
    text_y = y + pad

    draw.text((text_x, text_y), "LocusBlend export", font=title_font, fill=text)
    _, title_h = _text_size(draw, "LocusBlend export", title_font)
    context_text = _format_export_context(context)
    if context_text:
        _draw_wrapped_text(
            draw=draw,
            text=context_text,
            xy=(text_x, text_y + title_h + max(6, pad // 3)),
            max_width=max(1, text_w),
            font=body_font,
            fill=muted,
            line_spacing=max(3, h // 70),
        )

    if legend_imgs:
        legend_x = x + w - pad - legend_w
        legend_y = y + pad
        legend_h = max(1, h - 2 * pad)
        item_gap = max(6, legend_w // 36)
        item_w = max(1, (legend_w - item_gap * (len(legend_imgs) - 1)) // len(legend_imgs))
        for i, img in enumerate(legend_imgs):
            item_x = legend_x + i * (item_w + item_gap)
            paste_contained(canvas, img, (item_x, legend_y, item_w, legend_h))


def build_locusblend_export_image(
    locus_fig,
    compare_fig,
    export_context=None,
    output_width_in=8.5,
    output_height_in=11.0,
    dpi=300,
    include_legends=True,
    asset_dir=None,
):
    Image, ImageDraw, _ = _load_pillow()

    output_width_in = float(output_width_in)
    output_height_in = float(output_height_in)
    dpi = int(dpi)
    if not (3 <= output_width_in <= 40 and 3 <= output_height_in <= 40):
        raise ValueError("Export page dimensions must be between 3 and 40 inches.")
    if not (72 <= dpi <= 600):
        raise ValueError("Export DPI must be between 72 and 600.")

    canvas_w = int(output_width_in * dpi)
    canvas_h = int(output_height_in * dpi)
    if canvas_w * canvas_h > 80_000_000:
        raise ValueError("The requested export is too large. Reduce page size or DPI and try again.")

    canvas = Image.new("RGB", (canvas_w, canvas_h), "#ffffff")
    draw = ImageDraw.Draw(canvas)

    margin = max(30, int(0.35 * dpi))
    gap = max(16, int(0.12 * dpi))
    usable_w = canvas_w - 2 * margin
    usable_h = canvas_h - 2 * margin
    if usable_w <= 0 or usable_h <= 0:
        raise ValueError("Export page is too small for the requested margins.")

    header_h = int(1.15 * dpi) if include_legends else int(0.55 * dpi)
    header_h = max(header_h, 170 if include_legends else 95)
    header_h = min(header_h, max(60, int(usable_h * 0.34)))

    header_box = (margin, margin, usable_w, header_h)
    draw_export_header(
        canvas,
        draw,
        export_context or {},
        include_legends,
        header_box,
        asset_dir=asset_dir,
    )

    plot_top = margin + header_h + gap
    plot_h = canvas_h - plot_top - margin
    if plot_h <= gap + 2:
        raise ValueError("Export page is too small for both cached plots.")

    compare_mode = str((export_context or {}).get("compare_mode", ""))
    if compare_mode == "Single blended compare plot":
        locus_h = int(plot_h * 0.64)
    else:
        locus_h = int(plot_h * 0.62)
    compare_h = plot_h - locus_h - gap
    locus_h = max(1, locus_h)
    compare_h = max(1, compare_h)

    locus_png = render_plotly_figure_to_png_bytes(locus_fig, usable_w, locus_h, scale=1)
    locus_img = open_pil_image_from_bytes(locus_png)
    locus_box = (margin, plot_top, usable_w, locus_h)
    paste_contained(canvas, locus_img, locus_box)

    compare_y = plot_top + locus_h + gap
    compare_box = (margin, compare_y, usable_w, compare_h)
    if compare_mode == "Single blended compare plot":
        compare_render_size = max(1, min(usable_w, compare_h))
        compare_png = render_plotly_figure_to_png_bytes(
            compare_fig,
            compare_render_size,
            compare_render_size,
            scale=1,
        )
    else:
        compare_png = render_plotly_figure_to_png_bytes(compare_fig, usable_w, compare_h, scale=1)
    compare_img = open_pil_image_from_bytes(compare_png)
    paste_contained(canvas, compare_img, compare_box)

    return canvas.convert("RGB")


def image_to_export_bytes(canvas, output_format, dpi=300, filename_stem="locusblend_export"):
    """Serialize a composed Pillow canvas to PNG/PDF bytes.

    Returns ``(data, mime_type, filename)``. This is the reusable part of the
    app's ``make_locusblend_export_bytes``; that wrapper (which reads the cached
    figures from Streamlit session state and feeds a download button) stays in
    the Streamlit application.
    """
    if canvas is None:
        raise ValueError("No export canvas is available.")

    buf = BytesIO()
    fmt = str(output_format).upper()

    if fmt == "PNG":
        canvas.save(buf, format="PNG")
        return buf.getvalue(), "image/png", f"{filename_stem}.png"

    if fmt == "PDF":
        canvas.convert("RGB").save(buf, format="PDF", resolution=int(dpi))
        return buf.getvalue(), "application/pdf", f"{filename_stem}.pdf"

    raise ValueError(f"Unsupported export format: {output_format}")
