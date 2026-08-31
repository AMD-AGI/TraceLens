/**
 * TraceLens Model Explorer viewer.
 *
 * Expects a JSON payload produced by visualize_model_in_explorer.py and passed via
 * ?graph=<filename> (copied next to this page by the local server).
 */

if (window.modelExplorer) {
  window.modelExplorer.assetFilesBaseUrl =
    "https://unpkg.com/ai-edge-model-explorer-visualizer@0.1.2/dist/static_files";
  window.modelExplorer.workerScriptPath = "./worker.js";
}

const status = document.getElementById("status");
const graphPane = document.getElementById("tracelens-graph-pane");
const factSheet = document.getElementById("tracelens-fact-sheet");
const factSheetTitle = document.getElementById("tracelens-fact-sheet-title");
const factSheetBody = document.getElementById("tracelens-fact-sheet-body");

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
  factSheetBody.textContent = factSheetData.body;
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
  "Layer node attributes": { selected: false },
};

const TRACE_LENS_SHOW_ON_EDGE = {
  type: "Input metadata",
  filterText: "port_label",
};

function configureModelExplorerDisplay() {
  for (const legacyKey of LEGACY_SHOW_ON_NODE_KEYS) {
    localStorage.removeItem(legacyKey);
  }

  // Always apply TraceLens defaults so stale ME settings (e.g. "Tensor shape"
  // on edges, which renders "?" without metadata) do not override us.
  localStorage.setItem(SHOW_ON_NODE_KEY, JSON.stringify(TRACE_LENS_SHOW_ON_NODE));
  localStorage.setItem(SHOW_ON_EDGE_KEY, JSON.stringify(TRACE_LENS_SHOW_ON_EDGE));
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
