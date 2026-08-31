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

function mountVisualizer(graphCollections) {
  if (!localStorage.getItem("model_explorer_show_on_edge_item_v3")) {
    localStorage.setItem(
      "model_explorer_show_on_edge_item_v3",
      JSON.stringify({ type: "Input metadata", filterText: "port_label" }),
    );
  }

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
