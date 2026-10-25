# Vulkan runtime for llama.cpp on this Strix Point box.
# RADV_PERFTEST=nogttspill keeps weights in the dedicated VRAM pool
# instead of spilling them into GTT. VRAM and GTT here are one physical
# memory pool (32 GiB reported), not two stacks of RAM.

export AMD_VULKAN_ICD=RADV
export RADV_PERFTEST=nogttspill
export GGML_VK_VISIBLE_DEVICES="${GGML_VK_VISIBLE_DEVICES:-0}"

LLAMA_DIR="${LLAMA_DIR:-/home/vlad/opt/llama-b11685}"
export LD_LIBRARY_PATH="${LLAMA_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
