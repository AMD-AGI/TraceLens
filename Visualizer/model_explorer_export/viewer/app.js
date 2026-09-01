/**
 * TraceLens Model Explorer viewer.
 *
 * Expects a JSON payload produced by visualize_model_in_explorer.py and passed via
 * ?graph=<filename> (copied next to this page by the local server).
 */

function resolveWorkerScriptPath() {
  const embedded = document.getElementById("tracelens-worker-source");
  if (embedded?.textContent) {
    const blob = new Blob([embedded.textContent], { type: "application/javascript" });
    return URL.createObjectURL(blob);
  }
  return "./worker.js";
}

if (window.modelExplorer) {
  window.modelExplorer.assetFilesBaseUrl =
    "https://unpkg.com/ai-edge-model-explorer-visualizer@0.1.2/dist/static_files";
  window.modelExplorer.workerScriptPath = resolveWorkerScriptPath();
}

const status = document.getElementById("status");
const graphPane = document.getElementById("tracelens-graph-pane");
const factSheet = document.getElementById("tracelens-fact-sheet");
const factSheetTitle = document.getElementById("tracelens-fact-sheet-title");
const factSheetBody = document.getElementById("tracelens-fact-sheet-body");
const factSheetResizer = document.getElementById("tracelens-fact-sheet-resizer");

const FACT_SHEET_WIDTH_KEY = "tracelens_fact_sheet_width";
const DEFAULT_FACT_SHEET_WIDTH = 420;
const MIN_FACT_SHEET_WIDTH = 240;
const MAX_FACT_SHEET_WIDTH_RATIO = 0.6;

function clampFactSheetWidth(width) {
  const maxWidth = Math.max(
    MIN_FACT_SHEET_WIDTH,
    Math.floor(window.innerWidth * MAX_FACT_SHEET_WIDTH_RATIO),
  );
  return Math.min(maxWidth, Math.max(MIN_FACT_SHEET_WIDTH, Math.round(width)));
}

function setFactSheetWidth(width) {
  const clamped = clampFactSheetWidth(width);
  document.body.style.setProperty("--tracelens-fact-sheet-width", `${clamped}px`);
  try {
    localStorage.setItem(FACT_SHEET_WIDTH_KEY, String(clamped));
  } catch (_error) {
    // Ignore storage failures in restricted contexts.
  }
  return clamped;
}

function restoreFactSheetWidth() {
  try {
    const stored = localStorage.getItem(FACT_SHEET_WIDTH_KEY);
    if (stored) {
      const parsed = Number.parseInt(stored, 10);
      if (!Number.isNaN(parsed)) {
        setFactSheetWidth(parsed);
      }
    }
  } catch (_error) {
    // Ignore storage failures in restricted contexts.
  }
}

function initFactSheetResize() {
  if (!factSheetResizer) {
    return;
  }

  restoreFactSheetWidth();

  let dragging = false;

  const stopDragging = () => {
    if (!dragging) {
      return;
    }
    dragging = false;
    document.body.classList.remove("tracelens-fact-sheet-resizing");
  };

  factSheetResizer.addEventListener("mousedown", (event) => {
    event.preventDefault();
    dragging = true;
    document.body.classList.add("tracelens-fact-sheet-resizing");
  });

  window.addEventListener("mousemove", (event) => {
    if (!dragging) {
      return;
    }
    setFactSheetWidth(window.innerWidth - event.clientX);
  });

  window.addEventListener("mouseup", stopDragging);
  window.addEventListener("blur", stopDragging);

  factSheetResizer.addEventListener("keydown", (event) => {
    const step = event.shiftKey ? 40 : 16;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      const current = factSheet.getBoundingClientRect().width;
      setFactSheetWidth(current + step);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      const current = factSheet.getBoundingClientRect().width;
      setFactSheetWidth(current - step);
    }
  });

  window.addEventListener("resize", () => {
    setFactSheetWidth(factSheet.getBoundingClientRect().width);
  });
}

initFactSheetResize();

function showError(message) {
  status.textContent = message;
  status.hidden = false;
  graphPane.hidden = true;
  factSheet.hidden = true;
}

function graphParam() {
  const params = new URLSearchParams(window.location.search);
  return params.get("graph");
}

function loadEmbeddedPayload() {
  const embedded = document.getElementById("tracelens-payload");
  if (!embedded?.textContent) {
    return null;
  }
  return JSON.parse(embedded.textContent);
}

async function loadPayload() {
  const embedded = loadEmbeddedPayload();
  if (embedded) {
    return embedded;
  }

  const graphFile = graphParam();
  if (!graphFile) {
    throw new Error(
      "Missing graph payload. Serve via visualize_model_in_explorer.py --serve or pass ?graph=<filename>.",
    );
  }
  const response = await fetch(`./${encodeURIComponent(graphFile)}`);
  if (!response.ok) {
    throw new Error(`Failed to load ${graphFile}: HTTP ${response.status}`);
  }
  return response.json();
}

function mountFactSheet(factSheetData) {
  if (!factSheetData?.body) {
    factSheetTitle.textContent = "Fact sheet";
    factSheetBody.textContent = "No fact sheet data in this export.";
    return;
  }

  factSheetTitle.textContent = factSheetData.title || "Fact sheet";
  if (factSheetData.bodyHtml) {
    factSheetBody.innerHTML = factSheetData.bodyHtml;
  } else {
    factSheetBody.textContent = factSheetData.body;
  }
}

const SHOW_ON_NODE_KEY = "model_explorer_show_on_node_item_types_v2";
const SHOW_ON_EDGE_KEY = "model_explorer_show_on_edge_item_v3";
const LEGACY_SHOW_ON_NODE_KEYS = ["model_explorer_show_on_node_item_v3"];

// Model Explorer enum string values from ShowOnNodeItemType / ShowOnEdgeItemType.
const TRACE_LENS_SHOW_ON_NODE = {
  "Op node id": { selected: false },
  "Op node attributes": { selected: false, filterRegex: "output_shape" },
  "Op node inputs": { selected: false },
  "Op node outputs": { selected: false },
  "Layer node children count": { selected: false },
  "Layer node descendants count": { selected: false },
  // Model Explorer leaves an edge unlabeled when either end is a collapsed group, so
  // expandable blocks carry their boundary shapes as layer attributes instead.
  "Layer node attributes": { selected: true, filterRegex: "shape" },
};

const TRACE_LENS_SHOW_ON_EDGE = {
  type: "Tensor shape",
};

// Bump when the defaults below change so one refresh picks them up; between bumps the
// viewer keeps whatever the user selected in the Model Explorer display menu.
const DEFAULTS_VERSION_KEY = "tracelens_display_defaults_version";
const DEFAULTS_VERSION = "3";

function configureModelExplorerDisplay() {
  for (const legacyKey of LEGACY_SHOW_ON_NODE_KEYS) {
    localStorage.removeItem(legacyKey);
  }

  if (localStorage.getItem(DEFAULTS_VERSION_KEY) === DEFAULTS_VERSION) {
    return;
  }
  localStorage.setItem(SHOW_ON_NODE_KEY, JSON.stringify(TRACE_LENS_SHOW_ON_NODE));
  localStorage.setItem(SHOW_ON_EDGE_KEY, JSON.stringify(TRACE_LENS_SHOW_ON_EDGE));
  localStorage.setItem(DEFAULTS_VERSION_KEY, DEFAULTS_VERSION);
}

function mountVisualizer(graphCollections) {
  configureModelExplorerDisplay();

  const visualizer = document.createElement("model-explorer-visualizer");
  visualizer.graphCollections = graphCollections;
  visualizer.config = {
    showHorizontalScrollButton: true,
    showVerticalScrollButton: true,
    hideInfoPanel: true,
  };
  graphPane.replaceChildren(visualizer);
}

function showApp() {
  status.hidden = true;
  graphPane.hidden = false;
  factSheet.hidden = false;
}

loadPayload()
  .then((payload) => {
    const collections = payload.graphCollections;
    if (!Array.isArray(collections) || collections.length === 0) {
      throw new Error("JSON payload is missing graphCollections.");
    }
    mountFactSheet(payload.tracelensViewer?.factSheet);
    mountVisualizer(collections);
    showApp();
  })
  .catch((error) => {
    showError(error instanceof Error ? error.message : String(error));
  });
