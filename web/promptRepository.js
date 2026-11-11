import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "LoboForgePromptRepository";

function findWidget(node, name) {
    return (node.widgets || []).find((widget) => widget.name === name);
}

async function refreshPromptChoices(node) {
    const pathWidget = findWidget(node, "db_path");
    const promptWidget = findWidget(node, "prompt");
    if (!pathWidget || !promptWidget || !pathWidget.value) {
        return;
    }
    let data;
    try {
        const response = await api.fetchApi("/loboforge/prompt_repository/list", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ db_path: pathWidget.value }),
        });
        data = await response.json();
    } catch (err) {
        alert("Prompt list request failed: " + err);
        return;
    }
    if (data.error) {
        alert(data.error);
        return;
    }
    const choices = data.choices || [];
    if (!promptWidget.options) {
        promptWidget.options = {};
    }
    promptWidget.options.values = choices.length ? choices : ["(no prompts)"];
    if (choices.length && !choices.includes(promptWidget.value)) {
        promptWidget.value = choices[0];
    }
    app.graph.setDirtyCanvas(true, true);
}

async function chooseDatabase(node) {
    let data;
    try {
        const response = await api.fetchApi("/loboforge/prompt_repository/pick_db", { method: "POST" });
        data = await response.json();
    } catch (err) {
        alert("Database dialog request failed: " + err);
        return;
    }
    if (!data || data.cancelled) {
        return;
    }
    if (data.error) {
        alert(data.error);
        return;
    }
    const widget = findWidget(node, "db_path");
    if (!widget || !data.path) {
        return;
    }
    widget.value = data.path;
    if (widget.callback) {
        widget.callback(data.path);
    }
    await refreshPromptChoices(node);
}

app.registerExtension({
    name: "LoboForge.PromptRepository",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const hasButton = (this.widgets || []).some((widget) => widget.type === "button" && widget.name === "choose database");
            if (!hasButton) {
                const button = this.addWidget("button", "choose database", null, () => chooseDatabase(this));
                button.serialize = false;
            }
            const pathWidget = findWidget(this, "db_path");
            if (pathWidget) {
                const previous = pathWidget.callback;
                pathWidget.callback = (value) => {
                    if (previous) {
                        previous.call(this, value);
                    }
                    refreshPromptChoices(this);
                };
            }
            const size = this.computeSize();
            this.setSize([Math.max(this.size[0], size[0]), Math.max(this.size[1], size[1])]);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            refreshPromptChoices(this);
            return result;
        };
    },
});
