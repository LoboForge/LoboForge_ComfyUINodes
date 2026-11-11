import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAMES = new Set([
    "LoboForgeVideoPreview",
    "LoboForgePreviewImageAudio",
    "LoboForgeSaveClipAndLastFrame",
]);
const MAX_PREVIEW_HISTORY = 24;

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existant object");
        return;
    }
    if (property in object) {
        const callback_orig = object[property];
        object[property] = function () {
            const result = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return result;
        };
    } else {
        object[property] = callback;
    }
}

function buildViewParams(file, type, element, subfolder = "") {
    const params = {
        filename: file,
        type: type,
    };
    if (subfolder) {
        params.subfolder = subfolder;
    }
    let target_width = 256;
    if (element?.style?.width) {
        target_width = element.style.width.slice(0, -2) * 2;
    }
    if (!params.force_size || params.force_size.includes("?") || params.force_size == "Disabled") {
        params.force_size = target_width + "x?";
    } else {
        const size = params.force_size.split("x");
        const ar = parseInt(size[0]) / parseInt(size[1]);
        params.force_size = target_width + "x" + (target_width / ar);
    }
    return params;
}

function updateCarouselControls(previewWidget) {
    const count = previewWidget.history.length;
    const index = previewWidget.currentIndex;
    previewWidget.prevBtn.disabled = index >= count - 1;
    previewWidget.nextBtn.disabled = index <= 0;
    previewWidget.counterEl.textContent = count <= 1
        ? "Latest"
        : `${index + 1} / ${count}${index === 0 ? " · latest" : ""}`;
    previewWidget.navEl.hidden = count <= 1;
}

function showPreviewAt(previewWidget, previewNode, index) {
    const entry = previewWidget.history[index];
    if (!entry) {
        return;
    }
    previewWidget.currentIndex = index;
    previewWidget.value.params = entry.params;
    previewWidget.parentEl.hidden = previewWidget.value.hidden;
    // Never autoplay — user must hit play (avoids surprise audio on execute).
    previewWidget.videoEl.autoplay = false;
    previewWidget.videoEl.pause();
    previewWidget.videoEl.src = api.apiURL("/view?" + new URLSearchParams(entry.params));
    previewWidget.videoEl.hidden = false;
    updateCarouselControls(previewWidget);
    fitHeight(previewNode);
}

function ensurePreviewWidget(node) {
    let previewWidget = node.widgets?.find((widget) => widget.name === "videopreview");
    if (previewWidget?.videoEl) {
        return previewWidget;
    }

    const element = document.createElement("div");
    const previewNode = node;

    previewWidget = node.addDOMWidget("videopreview", "preview", element, {
        serialize: false,
        hideOnZoom: false,
        getValue() {
            return element.value;
        },
        setValue(value) {
            element.value = value;
        },
    });

    previewWidget.element = element;
    previewWidget.previewNode = previewNode;
    previewWidget.history = [];
    previewWidget.currentIndex = 0;
    previewWidget.computeSize = function (width) {
        if (this.aspectRatio && !this.parentEl.hidden) {
            let height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
            if (!(height > 0)) {
                height = 0;
            }
            this.computedHeight = height + 10;
            return [width, height];
        }
        return [width, -4];
    };

    previewWidget.value = { hidden: false, paused: true, params: {} };

    previewWidget.parentEl = document.createElement("div");
    previewWidget.parentEl.className = "video_preview";
    previewWidget.parentEl.style.width = "100%";
    element.appendChild(previewWidget.parentEl);

    previewWidget.navEl = document.createElement("div");
    previewWidget.navEl.className = "video_preview_nav";
    previewWidget.navEl.style.display = "flex";
    previewWidget.navEl.style.alignItems = "center";
    previewWidget.navEl.style.gap = "6px";
    previewWidget.navEl.style.marginBottom = "4px";
    previewWidget.navEl.hidden = true;

    previewWidget.prevBtn = document.createElement("button");
    previewWidget.prevBtn.type = "button";
    previewWidget.prevBtn.textContent = "◀";
    previewWidget.prevBtn.title = "Older preview";
    previewWidget.prevBtn.style.flex = "0 0 auto";

    previewWidget.counterEl = document.createElement("span");
    previewWidget.counterEl.style.flex = "1 1 auto";
    previewWidget.counterEl.style.textAlign = "center";
    previewWidget.counterEl.style.fontSize = "11px";
    previewWidget.counterEl.style.opacity = "0.85";

    previewWidget.nextBtn = document.createElement("button");
    previewWidget.nextBtn.type = "button";
    previewWidget.nextBtn.textContent = "▶";
    previewWidget.nextBtn.title = "Newer preview";
    previewWidget.nextBtn.style.flex = "0 0 auto";

    previewWidget.navEl.appendChild(previewWidget.prevBtn);
    previewWidget.navEl.appendChild(previewWidget.counterEl);
    previewWidget.navEl.appendChild(previewWidget.nextBtn);
    previewWidget.parentEl.appendChild(previewWidget.navEl);

    previewWidget.videoEl = document.createElement("video");
    previewWidget.videoEl.controls = true;
    previewWidget.videoEl.loop = false;
    previewWidget.videoEl.muted = false;
    previewWidget.videoEl.style.width = "100%";
    previewWidget.videoEl.addEventListener("loadedmetadata", () => {
        previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
        fitHeight(previewNode);
    });
    previewWidget.videoEl.addEventListener("error", () => {
        previewWidget.parentEl.hidden = true;
        fitHeight(previewNode);
    });
    previewWidget.parentEl.appendChild(previewWidget.videoEl);

    previewWidget.prevBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (previewWidget.currentIndex < previewWidget.history.length - 1) {
            showPreviewAt(previewWidget, previewNode, previewWidget.currentIndex + 1);
        }
    });
    previewWidget.nextBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (previewWidget.currentIndex > 0) {
            showPreviewAt(previewWidget, previewNode, previewWidget.currentIndex - 1);
        }
    });

    return previewWidget;
}

function previewVideo(node, file, type, subfolder = "") {
    const previewWidget = ensurePreviewWidget(node);
    const params = buildViewParams(file, type, previewWidget.element, subfolder);
    const entry = { file, type, subfolder, params: { ...params } };

    if (
        previewWidget.history[0]?.file === file &&
        previewWidget.history[0]?.type === type &&
        (previewWidget.history[0]?.subfolder || "") === (subfolder || "")
    ) {
        showPreviewAt(previewWidget, previewWidget.previewNode, 0);
        return;
    }

    previewWidget.history.unshift(entry);
    if (previewWidget.history.length > MAX_PREVIEW_HISTORY) {
        previewWidget.history.length = MAX_PREVIEW_HISTORY;
    }
    showPreviewAt(previewWidget, previewWidget.previewNode, 0);
}

function previewFromExecuted(node, data) {
    // Prefer LoboForge legacy ui.video so Comfy's native PreviewVideo path is unused
    // (returning images+animated would render a second player on the same node).
    if (data?.video?.length >= 2) {
        previewVideo(node, data.video[0], data.video[1], data.subfolder || "");
        return;
    }
    // Fallback: PreviewVideo contract images[{filename,subfolder,type}] + animated
    const image = data?.images?.[0];
    if (image?.filename) {
        previewVideo(node, image.filename, image.type || "output", image.subfolder || "");
    }
}

function addPreviewOptions(nodeType) {
    chainCallback(nodeType.prototype, "getExtraMenuOptions", function (_, options) {
        const optNew = [];
        try {
            const previewWidget = this.widgets.find((widget) => widget.name === "videopreview");
            const entry = previewWidget?.history?.[previewWidget.currentIndex ?? 0];
            let url = null;
            if (previewWidget?.videoEl?.hidden == false && previewWidget.videoEl.src) {
                url = previewWidget.videoEl.src;
            }
            if (url) {
                optNew.push(
                    {
                        content: "Open preview",
                        callback: () => {
                            window.open(url, "_blank");
                        },
                    },
                    {
                        content: "Save preview",
                        callback: () => {
                            const link = document.createElement("a");
                            link.href = url;
                            link.setAttribute("download", entry?.params?.filename ?? "preview.mp4");
                            document.body.append(link);
                            link.click();
                            requestAnimationFrame(() => link.remove());
                        },
                    }
                );
                if (previewWidget.history?.length > 1) {
                    optNew.push({
                        content: "Clear preview history",
                        callback: () => {
                            const latest = previewWidget.history[0];
                            previewWidget.history = latest ? [latest] : [];
                            previewWidget.currentIndex = 0;
                            updateCarouselControls(previewWidget);
                        },
                    });
                }
            }
            if (options.length > 0 && options[0] != null && optNew.length > 0) {
                optNew.push(null);
            }
            options.unshift(...optNew);
        } catch (error) {
            console.log(error);
        }
    });
}

app.registerExtension({
    name: "LoboForge.VideoPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData?.name;
        if (!NODE_NAMES.has(name)) {
            return;
        }
        // Do not set previewMediaType — that enables Comfy's native video preview
        // on top of our videopreview DOM widget (double player).
        nodeType.prototype.onExecuted = function (data) {
            previewFromExecuted(this, data);
        };
        addPreviewOptions(nodeType);
    },
});
