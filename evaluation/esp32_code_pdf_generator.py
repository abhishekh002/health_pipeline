"""
health_pipeline.evaluation.esp32_code_pdf_generator
Generates a simple, monochrome (black-and-white, zero-colour) PDF document
containing the complete ESP32 Arduino C++ firmware code and pinout specification.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import datetime


class MonochromeCodePDFBuilder:
    """
    Pure-Python PDF 1.4 generator creating clean, high-contrast,
    strictly monochrome (black & white) source code documentation.
    """

    PAGE_WIDTH = 595.28   # A4 width in points
    PAGE_HEIGHT = 841.89  # A4 height in points
    MARGIN_X = 40.0
    MARGIN_TOP = 45.0
    MARGIN_BOTTOM = 40.0

    def __init__(self, title: str = "ESP32 Firmware Source Code"):
        self.title = title
        self.pages_content: List[List[str]] = []
        self.current_page_ops: List[str] = []
        self.cursor_y = self.PAGE_HEIGHT - self.MARGIN_TOP
        self.current_page = 0
        self.new_page()

    def new_page(self):
        if self.current_page > 0:
            self.pages_content.append(self.current_page_ops)
        self.current_page += 1
        self.current_page_ops = []
        self.cursor_y = self.PAGE_HEIGHT - self.MARGIN_TOP
        self._add_running_header()

    def _add_running_header(self):
        # Top rule in pure black
        self.draw_line(
            self.MARGIN_X, self.PAGE_HEIGHT - 32,
            self.PAGE_WIDTH - self.MARGIN_X, self.PAGE_HEIGHT - 32,
            color=(0.0, 0.0, 0.0), width=0.8
        )
        self.draw_text(
            "ESP32 PATIENT HEALTH MONITOR  |  SOURCE CODE LISTING",
            self.MARGIN_X, self.PAGE_HEIGHT - 26,
            font="F2", size=8.0, color=(0.0, 0.0, 0.0)
        )
        self.draw_text(
            "PRINTABLE MONOCHROME SPECIFICATION",
            self.PAGE_WIDTH - self.MARGIN_X - 180, self.PAGE_HEIGHT - 26,
            font="F1", size=8.0, color=(0.2, 0.2, 0.2)
        )

    def check_page_break(self, needed_height: float):
        if self.cursor_y - needed_height < self.MARGIN_BOTTOM:
            self.new_page()

    def draw_line(
        self,
        x1: float, y1: float, x2: float, y2: float,
        color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        width: float = 1.0,
        dash: Optional[str] = None
    ):
        r, g, b = color
        d_op = f"[{dash}] 0 d\n" if dash else "[] 0 d\n"
        op = f"{d_op}{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S\n"
        self.current_page_ops.append(op)

    def draw_rect(
        self,
        x: float, y: float, w: float, h: float,
        fill: Optional[Tuple[float, float, float]] = None,
        stroke: Optional[Tuple[float, float, float]] = None,
        width: float = 1.0
    ):
        ops = []
        if fill:
            r, g, b = fill
            ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
        if stroke:
            r, g, b = stroke
            ops.append(f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w")
        
        ops.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re")
        if fill and stroke:
            ops.append("B\n")
        elif fill:
            ops.append("f\n")
        elif stroke:
            ops.append("S\n")
        else:
            ops.append("n\n")

        self.current_page_ops.append(" ".join(ops))

    def draw_text(
        self,
        text: str,
        x: float,
        y: float,
        font: str = "F1",
        size: float = 10.0,
        color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        clean = (
            text.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
        )
        r, g, b = color
        op = f"BT /{font} {size:.2f} Tf {r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} Td ({clean}) Tj ET\n"
        self.current_page_ops.append(op)

    def add_title_block(self, title: str, subtitle: str, metadata: str):
        self.check_page_break(75)
        # Clean white box with double black border
        box_w = self.PAGE_WIDTH - 2 * self.MARGIN_X
        box_h = 65.0
        self.draw_rect(self.MARGIN_X, self.cursor_y - box_h, box_w, box_h, fill=(1.0, 1.0, 1.0), stroke=(0.0, 0.0, 0.0), width=1.5)
        self.draw_rect(self.MARGIN_X + 2, self.cursor_y - box_h + 2, box_w - 4, box_h - 4, stroke=(0.0, 0.0, 0.0), width=0.5)

        self.draw_text(title, self.MARGIN_X + 14, self.cursor_y - 20, font="F2", size=13.0, color=(0.0, 0.0, 0.0))
        self.draw_text(subtitle, self.MARGIN_X + 14, self.cursor_y - 36, font="F1", size=9.0, color=(0.2, 0.2, 0.2))
        self.draw_text(metadata, self.MARGIN_X + 14, self.cursor_y - 52, font="F2", size=8.0, color=(0.0, 0.0, 0.0))

        self.cursor_y -= (box_h + 16)

    def add_section_title(self, title: str):
        self.check_page_break(32)
        box_w = self.PAGE_WIDTH - 2 * self.MARGIN_X
        self.draw_line(self.MARGIN_X, self.cursor_y, self.MARGIN_X + box_w, self.cursor_y, color=(0.0, 0.0, 0.0), width=1.0)
        self.draw_text(title.upper(), self.MARGIN_X + 2, self.cursor_y - 12, font="F2", size=10.0, color=(0.0, 0.0, 0.0))
        self.draw_line(self.MARGIN_X, self.cursor_y - 16, self.MARGIN_X + box_w, self.cursor_y - 16, color=(0.0, 0.0, 0.0), width=0.5)
        self.cursor_y -= 26

    def add_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        col_widths: List[float]
    ):
        table_w = sum(col_widths)
        row_h = 16.0
        head_h = 18.0
        
        self.check_page_break(head_h + row_h * len(rows) + 15)

        # Header Row (light gray background, black text and borders)
        self.draw_rect(self.MARGIN_X, self.cursor_y - head_h, table_w, head_h, fill=(0.92, 0.92, 0.92), stroke=(0.0, 0.0, 0.0), width=1.0)
        cur_x = self.MARGIN_X
        for idx, h in enumerate(headers):
            self.draw_text(h, cur_x + 5, self.cursor_y - 12, font="F2", size=8.5, color=(0.0, 0.0, 0.0))
            cur_x += col_widths[idx]
            if idx < len(headers) - 1:
                self.draw_line(cur_x, self.cursor_y, cur_x, self.cursor_y - head_h, color=(0.0, 0.0, 0.0), width=0.5)

        self.cursor_y -= head_h

        # Data Rows
        for r_idx, row in enumerate(rows):
            self.check_page_break(row_h)
            # Alternating white and very light gray
            bg = (1.0, 1.0, 1.0) if r_idx % 2 == 0 else (0.97, 0.97, 0.97)
            self.draw_rect(self.MARGIN_X, self.cursor_y - row_h, table_w, row_h, fill=bg, stroke=(0.0, 0.0, 0.0), width=0.5)
            
            cur_x = self.MARGIN_X
            for c_idx, cell in enumerate(row):
                font = "F3" if c_idx in (1, 2) else "F1"
                self.draw_text(str(cell), cur_x + 5, self.cursor_y - 11, font=font, size=8.0, color=(0.0, 0.0, 0.0))
                cur_x += col_widths[c_idx]
                if c_idx < len(row) - 1:
                    self.draw_line(cur_x, self.cursor_y, cur_x, self.cursor_y - row_h, color=(0.0, 0.0, 0.0), width=0.5)

            self.cursor_y -= row_h

        self.cursor_y -= 14

    def add_code_block(self, lines: List[str]):
        """
        Prints code with line numbers in Courier monospace font.
        Handles multi-page pagination cleanly with border frames.
        """
        line_height = 10.5

        for idx, line in enumerate(lines, start=1):
            self.check_page_break(line_height + 4)

            # Left gutter for line number
            num_str = f"{idx:03d} "
            self.draw_text(num_str, self.MARGIN_X + 4, self.cursor_y, font="F3", size=8.0, color=(0.4, 0.4, 0.4))
            
            # Gutter separator line
            self.draw_line(self.MARGIN_X + 28, self.cursor_y + 8, self.MARGIN_X + 28, self.cursor_y - 2, color=(0.8, 0.8, 0.8), width=0.4)

            # Code text (truncated to fit width if overly long)
            clean_line = line.replace("\t", "    ")
            if len(clean_line) > 85:
                clean_line = clean_line[:82] + "..."

            self.draw_text(clean_line, self.MARGIN_X + 34, self.cursor_y, font="F3", size=8.0, color=(0.0, 0.0, 0.0))
            self.cursor_y -= line_height

    def build_pdf_bytes(self) -> bytes:
        if self.current_page_ops:
            self.pages_content.append(self.current_page_ops)

        num_pages = len(self.pages_content)
        pdf_data = bytearray()
        pdf_data.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        offsets = []
        offsets.append(0)  # Object 0 dummy

        # 1. Catalog Object
        offsets.append(len(pdf_data))
        catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        pdf_data.extend(catalog)

        # 2. Pages Object
        page_obj_ids = [f"{3 + i * 2} 0 R" for i in range(num_pages)]
        kids_str = " ".join(page_obj_ids)
        offsets.append(len(pdf_data))
        pages_parent = f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>\nendobj\n".encode("latin1")
        pdf_data.extend(pages_parent)

        # Fonts: F1 (Helvetica), F2 (Helvetica-Bold), F3 (Courier)
        font_f1_id = 3 + num_pages * 2
        font_f2_id = font_f1_id + 1
        font_f3_id = font_f2_id + 1

        # Emit Page and Content objects
        for p_idx, page_ops in enumerate(self.pages_content):
            page_id = 3 + p_idx * 2
            content_id = page_id + 1

            # Running footer in pure black/gray
            footer_op = (
                f"BT /F1 8 Tf 0.0 0.0 0.0 rg {self.MARGIN_X:.2f} 24.00 Td "
                f"(ESP32 Arduino C++ Code Listing  |  Page {p_idx + 1} of {num_pages}) Tj ET\n"
                f"BT /F1 8 Tf 0.3 0.3 0.3 rg {self.PAGE_WIDTH - self.MARGIN_X - 160:.2f} 24.00 Td "
                f"(Black & White Printable Reference) Tj ET\n"
            )
            page_ops.append(footer_op)

            content_stream = "".join(page_ops).encode("latin1", errors="replace")
            content_len = len(content_stream)

            # Page Object
            offsets.append(len(pdf_data))
            page_obj = (
                f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.PAGE_WIDTH:.2f} {self.PAGE_HEIGHT:.2f}] "
                f"/Resources << /Font << /F1 {font_f1_id} 0 R /F2 {font_f2_id} 0 R /F3 {font_f3_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>\nendobj\n"
            ).encode("latin1")
            pdf_data.extend(page_obj)

            # Content Object
            offsets.append(len(pdf_data))
            content_obj = (
                f"{content_id} 0 obj\n<< /Length {content_len} >>\nstream\n".encode("latin1") +
                content_stream +
                b"\nendstream\nendobj\n"
            )
            pdf_data.extend(content_obj)

        # Emit Standard Font Dictionaries
        offsets.append(len(pdf_data))
        f1_dict = f"{font_f1_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n".encode("latin1")
        pdf_data.extend(f1_dict)

        offsets.append(len(pdf_data))
        f2_dict = f"{font_f2_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>\nendobj\n".encode("latin1")
        pdf_data.extend(f2_dict)

        offsets.append(len(pdf_data))
        f3_dict = f"{font_f3_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>\nendobj\n".encode("latin1")
        pdf_data.extend(f3_dict)

        # Cross-Reference Table
        xref_offset = len(pdf_data)
        total_objs = font_f3_id + 1
        xref_str = f"xref\n0 {total_objs}\n0000000000 65535 f \n"
        for off in offsets[1:]:
            xref_str += f"{off:010d} 00000 n \n"
        pdf_data.extend(xref_str.encode("latin1"))

        # Trailer
        trailer_str = f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        pdf_data.extend(trailer_str.encode("latin1"))

        return bytes(pdf_data)

    def save(self, filepath: Union[str, Path]) -> Path:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        pdf_bytes = self.build_pdf_bytes()
        with open(p, "wb") as f:
            f.write(pdf_bytes)
        return p


def generate_esp32_code_monochrome_pdf(
    source_code_path: Optional[Union[str, Path]] = None,
    output_pdf_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Generates a simple, clean, monochrome (zero-colour) PDF listing the complete
    ESP32 Arduino sketch, pinout connections, and compilation checklist.
    """
    src_file = Path(source_code_path or "D:/project/health_pipeline/esp32/esp32_health_monitor.ino")
    out_file = Path(output_pdf_path or "D:/project/health_pipeline/artifacts/reports/esp32_firmware_code.pdf")

    builder = MonochromeCodePDFBuilder(title="ESP32 Health Monitor Arduino C++ Code")

    # =========================================================================
    # PAGE 1: TITLE, WIRING TABLE, & CODE
    # =========================================================================
    builder.add_title_block(
        title="ESP32 PATIENT HEALTH MONITOR - ARDUINO C++ FIRMWARE",
        subtitle="Complete Source Code: AD8232 ECG + MAX30102 SpO2/HR + DS18B20 Temp + REST Telemetry",
        metadata="FORMAT: SIMPLE MONOCHROME PRINTABLE SPECIFICATION  |  TARGET: ESP32 DEVKIT V1"
    )

    builder.add_section_title("1. Hardware Pin Assignment & Bus Specification")
    headers = ["Sensor / Device", "Sensor Pin", "ESP32 GPIO Pin", "Bus / Protocol", "Electrical Notes"]
    rows = [
        ["AD8232 ECG Module", "OUTPUT", "GPIO 36 (VP)", "Analog ADC1_CH0", "12-bit ADC (0-4095), centered at 1.65V"],
        ["AD8232 ECG Module", "LO+ (Leads Off +)", "GPIO 2", "Digital Input", "Lead detachment detection (Active HIGH)"],
        ["AD8232 ECG Module", "LO- (Leads Off -)", "GPIO 4", "Digital Input", "Lead detachment detection (Active HIGH)"],
        ["MAX30102 Biosensor", "SDA", "GPIO 21", "I2C Data Bus", "Internal 3.3V logic with 400kHz fast mode"],
        ["MAX30102 Biosensor", "SCL", "GPIO 22", "I2C Clock Bus", "Connected with internal pull-ups"],
        ["DS18B20 Temp Probe", "DQ (Data)", "GPIO 15", "Dallas 1-Wire", "Requires 4.7 kOhm pull-up resistor to 3.3V"],
        ["Emergency Buzzer / LED", "Positive Anode", "GPIO 13", "Digital Output", "Active HIGH when CRITICAL triage alert fires"]
    ]
    builder.add_table(headers, rows, col_widths=[105.0, 95.0, 85.0, 95.0, 135.0])

    builder.add_section_title("2. Required Arduino Libraries")
    lib_headers = ["Library Name", "Author / Source", "Version", "Usage in Firmware"]
    lib_rows = [
        ["WiFi.h & HTTPClient.h", "Espressif Systems", "Built-in", "Wi-Fi connection & REST POST telemetry transmission"],
        ["Wire.h", "Arduino Core", "Built-in", "I2C communication with MAX30102 pulse oximeter"],
        ["SparkFun MAX3010x", "SparkFun Electronics", "v1.1.2+", "MAX30102 optical sensor setup & FIFO reading"],
        ["OneWire", "Paul Stoffregen", "v2.3.7+", "1-Wire bitbang communication protocol"],
        ["DallasTemperature", "Miles Burton", "v3.9.0+", "DS18B20 digital Celsius conversion"]
    ]
    builder.add_table(lib_headers, lib_rows, col_widths=[125.0, 115.0, 65.0, 210.0])

    builder.add_section_title("3. Complete Firmware Source Code (esp32_health_monitor.ino)")

    # Read the code lines
    code_text = src_file.read_text(encoding="utf-8")
    code_lines = code_text.splitlines()

    builder.add_code_block(code_lines)

    saved_path = builder.save(out_file)
    return saved_path


if __name__ == "__main__":
    p = generate_esp32_code_monochrome_pdf()
    print(f"Monochrome code PDF successfully created at: {p}")
