/* Web console logic: load model info, predict one row or a whole CSV. */

const state = { features: [], sample: {} };

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || response.statusText);
  }
  return response.json();
}

function showError(message) {
  const box = document.getElementById("error");
  box.textContent = message;
  box.hidden = !message;
}

/* Shorten the long column names from the dataset for display. */
function formatFeature(name) {
  return name
    .replace(/\(component \d+\)/gi, "")
    .replace(/\(kg in a m\^3 mixture\)/gi, "")
    .replace(/\(day\)/gi, "(days)")
    .replace(/\s+/g, " ")
    .trim();
}

function renderForm(sample) {
  const grid = document.getElementById("feature-grid");
  grid.innerHTML = "";

  state.features.forEach((name) => {
    const label = document.createElement("label");
    label.className = "field";

    const text = document.createElement("span");
    text.textContent = formatFeature(name);

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.required = true;
    input.dataset.feature = name;
    if (sample[name] !== undefined) input.value = sample[name];

    label.append(text, input);
    grid.append(label);
  });
}

function renderModel(info) {
  const box = document.getElementById("model-info");
  const chips = [
    ["Type", info.model_type],
    ["Target", "Strength"],
  ];

  if (info.metrics) {
    Object.entries(info.metrics).forEach(([name, value]) => {
      chips.push([name.toUpperCase(), Number(value).toFixed(3)]);
    });
  }

  box.innerHTML = chips
    .map(
      ([label, value]) =>
        `<span class="chip"><b>${label}</b>${value}</span>`
    )
    .join("");
}

async function predictSingle(event) {
  event.preventDefault();
  showError("");

  const row = {};
  document.querySelectorAll("input[data-feature]").forEach((input) => {
    row[input.dataset.feature] = Number(input.value);
  });

  try {
    const data = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: [row] }),
    });

    const box = document.getElementById("result");
    box.hidden = false;
    box.innerHTML = `<span class="value">${data.predictions[0].toFixed(2)}</span><span class="unit">MPa</span>`;
  } catch (error) {
    showError(error.message);
  }
}

/* Minimal CSV parser that also handles quoted fields. */
function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];

    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        value += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        value += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(value);
      value = "";
    } else if (char === "\n") {
      row.push(value);
      rows.push(row);
      row = [];
      value = "";
    } else if (char !== "\r") {
      value += char;
    }
  }

  if (value !== "" || row.length) {
    row.push(value);
    rows.push(row);
  }

  return rows.filter((cells) => cells.some((cell) => cell.trim() !== ""));
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeCsv(text) {
  const value = String(text);
  return /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

function renderBatch(header, rows, predictions) {
  const box = document.getElementById("batch-result");
  box.hidden = false;

  const visible = rows.slice(0, 20);
  const head = header.map((name) => `<th>${escapeHtml(name)}</th>`).join("");
  const body = visible
    .map(
      (cells, index) =>
        `<tr>${cells
          .map((cell) => `<td>${escapeHtml(cell)}</td>`)
          .join("")}<td class="pred">${predictions[index].toFixed(2)}</td></tr>`
    )
    .join("");

  const csv = [[...header, "prediction"], ...rows.map((cells, index) => [...cells, predictions[index]])]
    .map((line) => line.map(escapeCsv).join(","))
    .join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));

  box.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr>${head}<th>prediction</th></tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
    <p class="hint">Showing ${visible.length} of ${rows.length} rows.</p>
    <a class="primary link" href="${url}" download="predictions.csv">Download predictions</a>
  `;
}

async function predictBatch(event) {
  const file = event.target.files[0];
  if (!file) return;
  showError("");

  try {
    const table = parseCsv(await file.text());
    if (!table.length) throw new Error("CSV is empty");

    const header = table[0].map((cell) => cell.trim());
    const missing = state.features.filter((name) => !header.includes(name));
    if (missing.length) {
      throw new Error(`CSV is missing columns: ${missing.join(", ")}`);
    }

    const rows = table.slice(1).map((cells) => {
      const row = {};
      state.features.forEach((name) => {
        row[name] = Number(cells[header.indexOf(name)]);
      });
      return row;
    });
    if (!rows.length) throw new Error("CSV has no data rows");

    const data = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows }),
    });

    renderBatch(header, table.slice(1), data.predictions);
  } catch (error) {
    showError(error.message);
  }
}

async function init() {
  try {
    const [features, model, sample] = await Promise.all([
      api("/api/features"),
      api("/api/model"),
      api("/api/sample"),
    ]);

    state.features = features.features;
    state.sample = sample.sample;

    renderForm(state.sample);
    renderModel(model);
    document.getElementById("status").textContent =
      `Ready · ${state.features.length} features`;
  } catch (error) {
    document.getElementById("status").textContent = "Model not available";
    showError(error.message);
  }
}

document.getElementById("predict-form").addEventListener("submit", predictSingle);
document.getElementById("fill-sample").addEventListener("click", () => renderForm(state.sample));
document.getElementById("csv-input").addEventListener("change", predictBatch);

init();
