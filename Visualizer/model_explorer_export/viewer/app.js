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
const app = document.getElementById("tracelens-app");
const graphPane = document.getElementById("tracelens-graph-pane");
const factSheetTitle = document.getElementById("tracelens-fact-sheet-title");
const factSheetBody = document.getElementById("tracelens-fact-sheet-body");

function showError(message) {
  status.textContent = message;
  status.hidden = false;
  app.hidden = true;
}

function graphParam() {
  const params = new URLSearchParams(window.location.search);
  return params.get("graph");
}

async function loadPayload() {
  const graphFile = graphParam();
  if (!graphFile) {
    throw new Error("Missing ?graph=<filename> query parameter.");
  }
  const response = await fetch(`./${encodeURIComponent(graphFile)}`);
  if (!response.ok) {
    throw new Error(`Failed to load ${graphFile}: HTTP ${response.status}`);
  }
  return response.json();
}

function mountFactSheet(factSheet) {
  if (!factSheet?.body) {
    factSheetTitle.textContent = "Fact sheet";
    factSheetBody.textContent = "No fact sheet data in this export.";
    return;
  }

  factSheetTitle.textContent = factSheet.title || "Fact sheet";
  factSheetBody.textContent = factSheet.body;
}

function mountVisualizer(graphCollections) {
  const visualizer = document.createElement("model-explorer-visualizer");
  visualizer.graphCollections = graphCollections;
  visualizer.config = {
    showHorizontalScrollButton: true,
    showVerticalScrollButton: true,
  };
  graphPane.replaceChildren(visualizer);
}

function showApp() {
  status.hidden = true;
  app.hidden = false;
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
