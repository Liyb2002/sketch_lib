import { useEffect, useRef, useState } from "react";

// Minimal mask drawer:
// 1) Upload image
// 2) Hold mouse (or pen) to draw a freehand contour
// 3) Release to finalize: we close the contour and fill the inside
// 4) Export 3 PNGs (mask, overlay, cutout) all at same size as input

function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export default function App() {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  const [imgInfo, setImgInfo] = useState(null); // { name, w, h }
  const [isDrawing, setIsDrawing] = useState(false);
  const [points, setPoints] = useState([]); // raw points
  const [finalized, setFinalized] = useState(false);

  // Draw: image + contour preview
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background image (exact size)
    if (imgRef.current) {
      ctx.drawImage(imgRef.current, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = "#0b0f19";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#9aa4b2";
      ctx.font = "14px ui-sans-serif, system-ui";
      ctx.fillText("Upload an image to start", 16, 28);
      return;
    }

    // Contour stroke
    if (points.length >= 2) {
      ctx.save();
      ctx.strokeStyle = "#00A3FF";
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
      if (finalized) ctx.closePath();
      ctx.stroke();

      // Light fill preview after finalizing
      if (finalized && points.length >= 3) {
        ctx.globalAlpha = 0.2;
        ctx.fillStyle = "#00A3FF";
        ctx.fill();
      }

      ctx.restore();
    }
  }, [points, finalized, imgInfo]);

  function setCanvasToImageSize(imgEl) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = imgEl.naturalWidth;
    canvas.height = imgEl.naturalHeight;
  }

  function handleFile(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      setCanvasToImageSize(img);
      setImgInfo({ name: file.name, w: img.naturalWidth, h: img.naturalHeight });
      setPoints([]);
      setFinalized(false);
    };
    img.src = url;
  }

  function getCanvasPointFromEvent(ev) {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((ev.clientY - rect.top) / rect.height) * canvas.height;
    return { x, y };
  }

  function onPointerDown(ev) {
    if (!imgRef.current) return;
    ev.preventDefault();

    const p = getCanvasPointFromEvent(ev);
    if (!p) return;

    setIsDrawing(true);
    setFinalized(false);
    setPoints([p]);

    if (ev.currentTarget.setPointerCapture) {
      ev.currentTarget.setPointerCapture(ev.pointerId);
    }
  }

  function onPointerMove(ev) {
    if (!isDrawing) return;
    ev.preventDefault();

    const p = getCanvasPointFromEvent(ev);
    if (!p) return;

    setPoints((prev) => {
      if (prev.length === 0) return [p];
      const last = prev[prev.length - 1];
      if (dist(last, p) < 1.5) return prev;
      return prev.concat([p]);
    });
  }

  function onPointerUp(ev) {
    if (!isDrawing) return;
    ev.preventDefault();
    setIsDrawing(false);
    setFinalized(true); // finalize on release
  }

  function clear() {
    setPoints([]);
    setFinalized(false);
  }

  function exportAll() {
    if (!imgRef.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!finalized || points.length < 3) {
      alert("Draw a contour (draw and release) first.");
      return;
    }

    const baseName = (imgInfo && imgInfo.name ? imgInfo.name : "image").replace(/\.[^.]+$/, "");

    // 1) MASK: black background + white filled polygon
    const mask = document.createElement("canvas");
    mask.width = canvas.width;
    mask.height = canvas.height;
    const mctx = mask.getContext("2d");
    if (!mctx) return;

    mctx.fillStyle = "black";
    mctx.fillRect(0, 0, mask.width, mask.height);

    mctx.fillStyle = "white";
    mctx.beginPath();
    mctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) mctx.lineTo(points[i].x, points[i].y);
    mctx.closePath();
    mctx.fill();

    const maskUrl = mask.toDataURL("image/png");

    // 2) OVERLAY: original + translucent fill + outline
    const overlay = document.createElement("canvas");
    overlay.width = canvas.width;
    overlay.height = canvas.height;
    const octx = overlay.getContext("2d");
    if (!octx) return;

    octx.drawImage(imgRef.current, 0, 0, overlay.width, overlay.height);

    octx.save();
    octx.globalAlpha = 0.35;
    octx.fillStyle = "#00A3FF";
    octx.beginPath();
    octx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) octx.lineTo(points[i].x, points[i].y);
    octx.closePath();
    octx.fill();
    octx.restore();

    octx.save();
    octx.strokeStyle = "#00A3FF";
    octx.lineWidth = 2;
    octx.lineJoin = "round";
    octx.lineCap = "round";
    octx.beginPath();
    octx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) octx.lineTo(points[i].x, points[i].y);
    octx.closePath();
    octx.stroke();
    octx.restore();

    const overlayUrl = overlay.toDataURL("image/png");

    // 3) CUTOUT: white background, clip polygon, draw original image inside
    const cutout = document.createElement("canvas");
    cutout.width = canvas.width;
    cutout.height = canvas.height;
    const cctx = cutout.getContext("2d");
    if (!cctx) return;

    cctx.fillStyle = "white";
    cctx.fillRect(0, 0, cutout.width, cutout.height);

    cctx.save();
    cctx.beginPath();
    cctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) cctx.lineTo(points[i].x, points[i].y);
    cctx.closePath();
    cctx.clip();
    cctx.drawImage(imgRef.current, 0, 0, cutout.width, cutout.height);
    cctx.restore();

    const cutoutUrl = cutout.toDataURL("image/png");

    // Download all 3
    downloadDataUrl(maskUrl, baseName + "_mask.png");
    downloadDataUrl(overlayUrl, baseName + "_overlay.png");
    downloadDataUrl(cutoutUrl, baseName + "_cutout.png");
  }

  return (
    <div style={{ minHeight: "100vh", background: "#060812", color: "#e6e9ef" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: 20 }}>
        <h2 style={{ margin: 0, fontSize: 18 }}>Minimal Mask Drawer</h2>
        <div style={{ marginTop: 8, color: "#9aa4b2", fontSize: 13, lineHeight: 1.4 }}>
          Upload → hold mouse to draw → release to fill inside → export (mask + overlay + cutout).
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 16, marginTop: 16 }}>
          <div style={{ background: "#0b0f19", border: "1px solid #1b2233", borderRadius: 12, padding: 14 }}>
            <div style={{ display: "grid", gap: 10 }}>
              <label style={{ display: "grid", gap: 6 }}>
                <div style={{ fontSize: 12, color: "#b7c0cc" }}>Input image</div>
                <input type="file" accept="image/*" onChange={handleFile} />
              </label>

              <div style={{ fontSize: 12, color: "#9aa4b2" }}>
                {imgInfo ? "Image: " + imgInfo.w + " × " + imgInfo.h : "No image loaded"}
              </div>

              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 6 }}>
                <button
                  onClick={clear}
                  style={{
                    padding: "8px 10px",
                    borderRadius: 10,
                    border: "1px solid #26314a",
                    background: "#0e1322",
                    color: "#e6e9ef",
                    cursor: "pointer",
                  }}
                >
                  Clear
                </button>
                <button
                  onClick={exportAll}
                  style={{
                    padding: "8px 10px",
                    borderRadius: 10,
                    border: "1px solid #1e7bb6",
                    background: "#0b2a3d",
                    color: "#e6e9ef",
                    cursor: "pointer",
                  }}
                >
                  Save mask (+ overlay + cutout)
                </button>
              </div>
            </div>
          </div>

          <div
            style={{
              background: "#0b0f19",
              border: "1px solid #1b2233",
              borderRadius: 12,
              padding: 14,
              overflow: "auto",
            }}
          >
            <div
              style={{
                display: "inline-block",
                borderRadius: 10,
                border: "1px solid #202a40",
                overflow: "hidden",
                background: "#000",
                touchAction: "none",
              }}
            >
              <canvas
                ref={canvasRef}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
                style={{
                  display: "block",
                  maxWidth: "100%",
                  height: "auto",
                  cursor: imgInfo ? "crosshair" : "not-allowed",
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
