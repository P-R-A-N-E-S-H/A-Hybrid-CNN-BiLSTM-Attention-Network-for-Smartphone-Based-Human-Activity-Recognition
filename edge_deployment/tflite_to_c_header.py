"""
======================================================================
tflite_to_c_header.py - Microcontroller C Header Array Exporter
======================================================================

Converts compiled .tflite model binary files into C/C++ source header files
containing `const unsigned char model_data[]` byte arrays for direct inclusion
in embedded firmware projects (ESP32, Arduino, ARM Cortex-M, STM32, Raspberry Pi Pico).

Author: Pranesh
======================================================================
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config


def tflite_to_c_array(
    tflite_path: str,
    output_header_path: str = None,
    array_name: str = "har_model_data"
) -> str:
    """Converts a binary .tflite file into a C header file (.h)."""
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite file not found at: {tflite_path}")

    if output_header_path is None:
        base_name = Path(tflite_path).stem
        output_header_path = str(Path(tflite_path).parent / f"{base_name}.h")

    with open(tflite_path, "rb") as f:
        bytes_data = f.read()

    total_len = len(bytes_data)

    header_lines = [
        "/*",
        " * ======================================================================",
        f" * Auto-Generated Embedded C Model Array: {Path(output_header_path).name}",
        f" * Source: {Path(tflite_path).name} ({total_len} bytes / {total_len/1024.0:.2f} KB)",
        " * Target: ESP32 / Arduino / ARM Cortex-M Embedded Deployment",
        " * Author: Pranesh",
        " * ======================================================================",
        " */",
        "",
        "#ifndef HAR_MODEL_DATA_H",
        "#define HAR_MODEL_DATA_H",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "// Memory alignment for microcontroller DMA/SIMD ops",
        f"alignas(16) const unsigned char {array_name}[] = {{"
    ]

    # Format bytes into rows of 12 hex values
    row = []
    for i, b in enumerate(bytes_data):
        row.append(f"0x{b:02x}")
        if len(row) == 12:
            header_lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        header_lines.append("    " + ", ".join(row))

    header_lines.extend([
        "};",
        "",
        f"const unsigned int {array_name}_len = {total_len};",
        "",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        "#endif // HAR_MODEL_DATA_H",
        ""
    ])

    with open(output_header_path, "w") as f:
        f.write("\n".join(header_lines))

    print(f"[C Exporter] Generated C Header: {output_header_path} ({total_len} bytes)")
    return output_header_path


if __name__ == "__main__":
    tflite_dir = os.path.join(config.BASE_DIR, "edge_deployment", "tflite_models")
    if os.path.exists(tflite_dir):
        for fname in os.listdir(tflite_dir):
            if fname.endswith(".tflite"):
                tpath = os.path.join(tflite_dir, fname)
                tflite_to_c_array(tpath)
