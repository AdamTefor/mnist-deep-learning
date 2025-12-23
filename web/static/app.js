const canvas = document.getElementById("pad");
const ctx = canvas.getContext("2d");

const btnPredict = document.getElementById("btnPredict");
const btnClear = document.getElementById("btnClear");

const predEl = document.getElementById("pred");
const confEl = document.getElementById("conf");
const top3El = document.getElementById("top3");

// Setup dessin
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.lineWidth = 18;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

let drawing = false;

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX ?? e.touches[0].clientX) - rect.left;
  const y = (e.clientY ?? e.touches[0].clientY) - rect.top;
  return { x, y };
}

function startDraw(e) {
  drawing = true;
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
  e.preventDefault();
}

function draw(e) {
  if (!drawing) return;
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
  e.preventDefault();
}

function endDraw() {
  drawing = false;
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseleave", endDraw);

// Mobile
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", endDraw);

btnClear.addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  predEl.textContent = "-";
  confEl.textContent = "-";
  top3El.innerHTML = "";
});

btnPredict.addEventListener("click", async () => {
  const imageDataUrl = canvas.toDataURL("image/png");

  predEl.textContent = "...";
  confEl.textContent = "...";
  top3El.innerHTML = "";

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageDataUrl })
  });

  const data = await res.json();
  if (data.error) {
    predEl.textContent = "Erreur";
    confEl.textContent = data.error;
    return;
  }

  predEl.textContent = data.prediction;
  confEl.textContent = (data.confidence * 100).toFixed(2) + "%";

  data.top3.forEach(item => {
    const li = document.createElement("li");
    li.textContent = `${item.digit} : ${(item.prob * 100).toFixed(2)}%`;
    top3El.appendChild(li);
  });
});
