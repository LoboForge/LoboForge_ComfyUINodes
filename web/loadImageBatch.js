import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "LoboForgeLoadImageBatch";

const WIDGET_LABELS = {
    folder: "Folder",
    pattern: "Pattern",
    how_many: "How many",
    order: "Order",
    seed: "Seed",
    start_at: "Start at",
    allow_RGBA_output: "Allow RGBA",
    filename_text_extension: "Keep filename extension",
    // Older widget names still present on stale graphs until reconfigured
    folder_path: "Folder",
    file_pattern: "Pattern",
    pick_order: "Order",
    path: "Folder",
};

function findWidget(node, name) {
    return (node.widgets || []).find((widget) => widget.name === name);
}

function applyWidgetLabels(node) {
    for (const widget of node.widgets || []) {
        const label = WIDGET_LABELS[widget.name];
        if (label) {
            widget.label = label;
        }
    }
}

function repairShiftedPattern(node) {
    const pattern = findWidget(node, "pattern") || findWidget(node, "file_pattern");
    if (pattern && (pattern.value === "true" || pattern.value === "false")) {
        pattern.value = "*";
    }
}

async function chooseBatchFolder(node) {
    let data;
    try {
        const response = await api.fetchApi("/loboforge/load_image_batch/pick_folder", {
            method: "POST",
        });
        data = await response.json();
    } catch (err) {
        alert("Folder dialog request failed: " + err);
        return;
    }
    if (!data || data.cancelled) {
        return;
    }
    if (data.error) {
        alert(data.error);
        return;
    }
    const widget =
        findWidget(node, "folder") ||
        findWidget(node, "folder_path") ||
        findWidget(node, "path");
    if (!widget || !data.path) {
        return;
    }
    widget.value = data.path;
    if (widget.callback) {
        widget.callback(data.path);
    }
    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "LoboForge.LoadImageBatch",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            repairShiftedPattern(this);
            applyWidgetLabels(this);
            const hasButton = (this.widgets || []).some(
                (widget) => widget.type === "button" && widget.name === "choose folder"
            );
            if (!hasButton) {
                const button = this.addWidget("button", "choose folder", null, () =>
                    chooseBatchFolder(this)
                );
                button.serialize = false;
            }
            const size = this.computeSize();
            this.setSize([
                Math.max(this.size[0], size[0]),
                Math.max(this.size[1], size[1]),
            ]);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            repairShiftedPattern(this);
            applyWidgetLabels(this);
            return result;
        };
    },
});
