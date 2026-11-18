# LoboForge

ComfyUI custom nodes: batch image/prompt loading, live previews, video I/O, prompt repository, MiniMax H3 turbo helpers, and thin replacements so `workflows/H3_Turbo_batchLoop_fixed.json` runs with **core ComfyUI + LoboForge only** (no other custom node packs).

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LoboForge/LoboForge_Comfyui_Nodes.git LoboForge
```

Restart ComfyUI (or reload custom nodes). No extra pip deps beyond ComfyUI itself.

## Folder → one video per image

`LoboForge Load Folder Images` returns an `OUTPUT_IS_LIST` fan-out (not one stacked batch tensor). MiniMax H3 `ReferenceToVideo` only uses `img[:1]` per ref socket — it does not iterate a batch dim.

For H3 folder batches:

1. Set **how_many=0** (all files) or N.
2. Prefer **filename_text_extension=false** (stem only).
3. Wire **LoboForge Save Video (Named)** to the encoded `VIDEO` + `filename_text` so each item writes `output/video/H3_batch/<stem>_#####_.mp4`.

See `workflows/H3_Turbo_batchLoop_fixed.json`.

## H3 batch workflow deps

That workflow uses built-in MiniMax-H3 / sampler / video nodes plus LoboForge batch/save/encode helpers. Community packs that used to be required (rgthree, KJNodes, Lokken, ControlFlowUtils, NVIDIA RTX, H3 PowerLoRA stack, community 3D latent upscaler) are replaced by core nodes or LoboForge stand-ins in `workflow_compat.py`.
