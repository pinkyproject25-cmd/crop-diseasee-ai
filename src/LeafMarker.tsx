import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

type Point = { x: number; y: number };
type Props = { imageUrl: string; onMeasured: (percentage: number) => void };

const BRUSH_WIDTH = 22; // At the canvas's maximum 640-pixel width.

function polygon(ctx: CanvasRenderingContext2D, points: Point[], width: number, height: number) {
  if (points.length === 0) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x * width, points[0].y * height);
  for (const point of points.slice(1)) ctx.lineTo(point.x * width, point.y * height);
  ctx.closePath();
}

function strokes(ctx: CanvasRenderingContext2D, lines: Point[][], width: number, height: number) {
  ctx.lineWidth = BRUSH_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const line of lines) {
    if (!line.length) continue;
    ctx.beginPath();
    ctx.moveTo(line[0].x * width, line[0].y * height);
    for (const point of line.slice(1)) ctx.lineTo(point.x * width, point.y * height);
    if (line.length === 1) ctx.lineTo(line[0].x * width + .01, line[0].y * height);
    ctx.stroke();
  }
}

export default function LeafMarker({ imageUrl, onMeasured }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const drawingRef = useRef(false);
  const [leafPoints, setLeafPoints] = useState<Point[]>([]);
  const [leafClosed, setLeafClosed] = useState(false);
  const [marks, setMarks] = useState<Point[][]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      if (cancelled || !canvasRef.current) return;
      imageRef.current = image;
      const width = Math.min(640, image.naturalWidth);
      canvasRef.current.width = width;
      canvasRef.current.height = Math.max(1, Math.round(width * image.naturalHeight / image.naturalWidth));
      setLeafPoints([]);
      setLeafClosed(false);
      setMarks([]);
    };
    image.src = imageUrl;
    return () => { cancelled = true; };
  }, [imageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !imageRef.current) return;
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(imageRef.current, 0, 0, width, height);
    if (leafPoints.length) {
      ctx.beginPath();
      ctx.moveTo(leafPoints[0].x * width, leafPoints[0].y * height);
      for (const point of leafPoints.slice(1)) ctx.lineTo(point.x * width, point.y * height);
      ctx.strokeStyle = "#33f1a2";
      ctx.lineWidth = 3;
      if (leafClosed) {
        ctx.closePath();
        ctx.fillStyle = "rgba(32, 218, 133, .15)";
        ctx.fill();
      }
      ctx.stroke();
      for (const point of leafPoints) {
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#00ff97";
        ctx.fill();
      }
    }
    ctx.strokeStyle = "rgba(255, 74, 45, .65)";
    strokes(ctx, marks, width, height);
  }, [imageUrl, leafPoints, leafClosed, marks]);

  function point(event: ReactPointerEvent<HTMLCanvasElement>): Point {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    };
  }

  function start(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!imageRef.current) return;
    setError("");
    const selectedPoint = point(event);
    if (!leafClosed) {
      setLeafPoints((previous) => [...previous, selectedPoint]);
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    drawingRef.current = true;
    setMarks((previous) => [...previous, [selectedPoint]]);
  }

  function move(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!leafClosed || !drawingRef.current) return;
    const next = point(event);
    setMarks((previous) => previous.map((line, index) =>
      index === previous.length - 1 ? [...line, next] : line,
    ));
  }

  function calculate() {
    const canvas = canvasRef.current;
    if (!canvas || leafPoints.length < 3 || !leafClosed || marks.length === 0) return;
    const leafCanvas = document.createElement("canvas");
    const markCanvas = document.createElement("canvas");
    leafCanvas.width = markCanvas.width = canvas.width;
    leafCanvas.height = markCanvas.height = canvas.height;
    const leafCtx = leafCanvas.getContext("2d");
    const markCtx = markCanvas.getContext("2d");
    if (!leafCtx || !markCtx) return;
    leafCtx.fillStyle = "white";
    polygon(leafCtx, leafPoints, canvas.width, canvas.height);
    leafCtx.fill();
    markCtx.strokeStyle = "white";
    strokes(markCtx, marks, canvas.width, canvas.height);
    const leaf = leafCtx.getImageData(0, 0, canvas.width, canvas.height).data;
    const damage = markCtx.getImageData(0, 0, canvas.width, canvas.height).data;
    let leafPixels = 0;
    let affectedPixels = 0;
    for (let i = 3; i < leaf.length; i += 4) {
      if (leaf[i] > 127) {
        leafPixels++;
        if (damage[i] > 127) affectedPixels++;
      }
    }
    if (leafPixels < 400 || affectedPixels < 20) {
      setError("Outline the visible leaf and paint the affected areas before calculating.");
      return;
    }
    const percentage = Math.round((affectedPixels / leafPixels) * 1000) / 10;
    if (percentage >= 100) {
      setError("Paint only the affected patches within the leaf, then retry.");
      return;
    }
    onMeasured(percentage);
  }

  return (
    <section className="report-panel" aria-label="User-assisted visible area measurement">
      <h3>Mark visible affected area</h3>
      <p>This optional prototype measurement uses your markings on the actual photo. It is not an automatic lesion diagnosis.</p>
      <ol>
        <li>Click around the outline of the visible leaf, excluding the fingers and background. Choose “Finish leaf outline.”</li>
        <li>Drag over the visibly affected patches. Choose “Calculate from marks.”</li>
      </ol>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label="Uploaded leaf photograph with user markings"
        style={{ display: "block", width: "100%", maxWidth: 640, height: "auto", borderRadius: 10, touchAction: "none", cursor: "crosshair" }}
        onPointerDown={start}
        onPointerMove={move}
        onPointerUp={() => { drawingRef.current = false; }}
        onPointerCancel={() => { drawingRef.current = false; }}
      />
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 14 }}>
        {!leafClosed && <button className="secondary-button" disabled={leafPoints.length < 3} onClick={() => setLeafClosed(true)}>Finish leaf outline</button>}
        {leafClosed && <button className="primary-button" disabled={!marks.length} onClick={calculate}>Calculate from marks</button>}
        {leafClosed && <button className="secondary-button" disabled={!marks.length} onClick={() => setMarks((previous) => previous.slice(0, -1))}>Undo last mark</button>}
        <button className="secondary-button" onClick={() => { setLeafPoints([]); setLeafClosed(false); setMarks([]); setError(""); }}>Start over</button>
      </div>
      {error && <p className="inline-error">{error}</p>}
    </section>
  );
}
