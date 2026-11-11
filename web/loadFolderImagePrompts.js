import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "LoboForgeLoadFolderImagePrompts";

const WIDGET_LABELS = {
    folder: "Folder",
    pattern: "Pattern",
    how_many: "How many",
    order: "Order",
    seed: "Seed",
    start_at: "Start at",
    skip_missing: "Skip missing",
    write_sidecar_txt: "Write sidecar txt",
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

async function chooseImagePromptsFolder(node) {
    let data;
    try {
        const response = await api.fetchApi("/loboforge/load_folder_image_prompts/pick_folder", {
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
    const widget = findWidget(node, "folder");
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
    name: "LoboForge.LoadFolderImagePrompts",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            applyWidgetLabels(this);
            const hasButton = (this.widgets || []).some(
                (widget) => widget.type === "button" && widget.name === "choose folder"
            );
            if (!hasButton) {
                const button = this.addWidget("button", "choose folder", null, () =>
                    chooseImagePromptsFolder(this)
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
            applyWidgetLabels(this);
            return result;
        };
    },
});
