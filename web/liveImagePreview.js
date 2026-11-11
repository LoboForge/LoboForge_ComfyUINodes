// Live preview images use the stock PreviewImage gallery/lightbox path on the
// leaf node. Mid-batch growth comes from intermediate executed events with
// merge=true (see live_preview_common.push_live_tiles).
//
// ComfyUI subgraph hosts do not show nested ui.images via nodeMedia; they only
// surface them through properties.previewExposures → usePromotedPreviews.
// Ensure that exposure exists, and also mirror a clickable gallery onto the
// parent SubgraphNode so previews are visible without diving in.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LIVE_CLASSES = new Set([
    "LoboForgeKSamplerLive",
    "LoboForgeSamplerCustomAdvancedLive",
    "LoboForgeLiveImagePreview",
]);

const PREVIEW_NAME = "$$canvas-image-preview";
const HOST_WIDGET = "loboforge_subgraph_preview";

function getNodeById(graph, id) {
    if (!graph?.getNodeById) {
        return null;
    }
    return (
        graph.getNodeById(id) ||
        graph.getNodeById(Number(id)) ||
        graph.getNodeById(String(id)) ||
        null
    );
}

function resolveChain(rootGraph, executionId) {
    const parts = String(executionId).split(":").filter(Boolean);
    if (parts.length < 2 || !rootGraph) {
        return null;
    }
    let graph = rootGraph;
    const chain = [];
    for (let i = 0; i < parts.length; i++) {
        const node = getNodeById(graph, parts[i]);
        if (!node) {
            return null;
        }
        chain.push(node);
        if (i < parts.length - 1) {
            if (!node.subgraph) {
                return null;
            }
            graph = node.subgraph;
        }
    }
    return chain;
}

function leafType(node) {
    return node?.type || node?.comfyClass || node?.constructor?.nodeData?.name || "";
}

function viewUrl(image) {
    const params = new URLSearchParams(image);
    return api.apiURL(`/view?${params}`);
}

function ensureExposure(host, sourceNodeId) {
    if (!host) {
        return;
    }
    const sid = String(sourceNodeId);
    host.properties = host.properties || {};
    const cur = Array.isArray(host.properties.previewExposures)
        ? host.properties.previewExposures
        : [];
    const exists = cur.some(
        (entry) =>
            String(entry.sourceNodeId) === sid &&
            entry.sourcePreviewName === PREVIEW_NAME
    );
    if (!exists) {
        host.properties.previewExposures = [
            ...cur,
            {
                name: PREVIEW_NAME,
                sourceNodeId: sid,
                sourcePreviewName: PREVIEW_NAME,
            },
        ];
    }
    // Sync into the frontend previewExposureStore when available.
    try {
        host._hydratePreviewExposures?.();
    } catch (_err) {
        // Private hydrate is best-effort across frontend versions.
    }
}

function openLightbox(urls, startIndex) {
    const existing = document.getElementById("loboforge-live-lightbox");
    if (existing) {
        existing.remove();
    }
    let index = Math.max(0, Math.min(startIndex, urls.length - 1));
    const overlay = document.createElement("div");
    overlay.id = "loboforge-live-lightbox";
    overlay.style.cssText =
        "position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,0.85);" +
        "display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px;";
    const img = document.createElement("img");
    img.style.cssText = "max-width:92vw;max-height:85vh;object-fit:contain;cursor:pointer;";
    const caption = document.createElement("div");
    caption.style.cssText = "color:#ddd;font:12px sans-serif;";
    const show = () => {
        img.src = urls[index];
        caption.textContent = `${index + 1} / ${urls.length}  (Esc to close, ←/→ to browse)`;
    };
    const close = () => {
        overlay.remove();
        window.removeEventListener("keydown", onKey);
    };
    const onKey = (event) => {
        if (event.key === "Escape") {
            close();
        } else if (event.key === "ArrowLeft") {
            index = (index - 1 + urls.length) % urls.length;
            show();
        } else if (event.key === "ArrowRight") {
            index = (index + 1) % urls.length;
            show();
        }
    };
    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) {
            close();
        }
    });
    img.addEventListener("click", (event) => {
        event.stopPropagation();
        index = (index + 1) % urls.length;
        show();
    });
    window.addEventListener("keydown", onKey);
    overlay.appendChild(img);
    overlay.appendChild(caption);
    document.body.appendChild(overlay);
    show();
}

function ensureHostGallery(host) {
    let widget = host.widgets?.find((entry) => entry.name === HOST_WIDGET);
    if (widget?.element) {
        return widget;
    }
    const el = document.createElement("div");
    el.className = "loboforge-subgraph-preview";
    el.style.cssText =
        "display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));" +
        "gap:4px;padding:6px;width:100%;min-height:128px;" +
        "background:rgba(0,0,0,0.35);border-radius:4px;box-sizing:border-box;";
    widget = host.addDOMWidget(HOST_WIDGET, "image", el, {
        serialize: false,
        hideOnZoom: false,
    });
    widget.element = el;
    widget.computeSize = function (width) {
        const count = Math.max(1, el.childElementCount);
        const cols = Math.max(1, Math.floor((width - 24) / 100));
        const rows = Math.ceil(count / cols);
        return [width, Math.max(148, rows * 100 + 20)];
    };
    return widget;
}

function setHostGallery(host, images, merge) {
    const widget = ensureHostGallery(host);
    const el = widget.element;
    if (!merge) {
        el.replaceChildren();
    }
    const existing = new Set(
        [...el.querySelectorAll("img")].map((img) => img.dataset.filename || img.src)
    );
    for (const image of images) {
        const key = image.filename || JSON.stringify(image);
        if (existing.has(key)) {
            continue;
        }
        existing.add(key);
        const url = viewUrl(image);
        const img = document.createElement("img");
        img.src = url;
        img.dataset.filename = image.filename || "";
        img.alt = image.filename || "preview";
        img.style.cssText =
            "width:100%;aspect-ratio:1;object-fit:contain;background:#111;" +
            "cursor:pointer;border-radius:2px;";
        img.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            const urls = [...el.querySelectorAll("img")].map((node) => node.src);
            openLightbox(urls, urls.indexOf(img.src));
        });
        el.appendChild(img);
    }
    try {
        host.setSize?.(host.computeSize?.() || host.size);
    } catch (_err) {
        // Size refresh is best-effort.
    }
    host.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function mirrorLeafOutputs(host, leaf, detail) {
    const subgraphId = host.subgraph?.id;
    if (!subgraphId || !app.nodeOutputs) {
        return;
    }
    const locator = `${subgraphId}:${leaf.id}`;
    const images = detail.output?.images || [];
    const prev = app.nodeOutputs[locator];
    if (detail.merge && prev?.images?.length) {
        app.nodeOutputs[locator] = {
            ...prev,
            ...detail.output,
            images: prev.images.concat(images),
        };
    } else {
        app.nodeOutputs[locator] = { ...(detail.output || {}), images: [...images] };
    }
    // Prefer decoded ui.images over latent blob previews for promotion.
    if (app.nodePreviewImages) {
        delete app.nodePreviewImages[locator];
        delete app.nodePreviewImages[String(detail.display_node || detail.node || "")];
    }
}

function onLiveExecuted(detail) {
    const executionId = String(detail.display_node || detail.node || "");
    const images = detail.output?.images;
    if (!images?.length || !executionId.includes(":")) {
        return;
    }
    // Must resolve from the root graph — app.graph is the open subgraph when dived-in.
    const root =
        app.rootGraph ||
        app.graph?.rootGraph ||
        app.canvas?.graph?.rootGraph ||
        app.graph;
    const chain = resolveChain(root, executionId);
    if (!chain || chain.length < 2) {
        return;
    }
    const leaf = chain[chain.length - 1];
    if (!LIVE_CLASSES.has(leafType(leaf))) {
        return;
    }
    // Update every enclosing subgraph host (outermost first).
    for (let i = 0; i < chain.length - 1; i++) {
        const host = chain[i];
        if (!host?.subgraph) {
            continue;
        }
        const sourceId = chain[i + 1].id;
        ensureExposure(host, sourceId);
        if (i === chain.length - 2) {
            mirrorLeafOutputs(host, leaf, detail);
        }
        setHostGallery(host, images, !!detail.merge);
    }
}

function ensureExposuresForGraph(graph) {
    if (!graph?.nodes) {
        return;
    }
    for (const node of graph.nodes) {
        if (!node.subgraph?.nodes) {
            continue;
        }
        for (const inner of node.subgraph.nodes) {
            if (LIVE_CLASSES.has(leafType(inner))) {
                ensureExposure(node, inner.id);
            }
            if (inner.subgraph) {
                ensureExposuresForGraph(inner.subgraph);
            }
        }
    }
}

app.registerExtension({
    name: "LoboForge.LiveImagePreview",

    async setup() {
        api.addEventListener("executed", ({ detail }) => {
            try {
                onLiveExecuted(detail);
            } catch (err) {
                console.warn("[LoboForge] subgraph preview bubble failed:", err);
            }
        });
    },

    async afterConfigureGraph() {
        try {
            ensureExposuresForGraph(app.graph);
        } catch (err) {
            console.warn("[LoboForge] preview exposure ensure failed:", err);
        }
    },
});
