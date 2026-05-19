# Digital Watermarking using DCT (Discrete Cosine Transform)

This project implements a digital watermarking system that embeds a binary watermark into a host image using the Discrete Cosine Transform (DCT) in the frequency domain. It includes tools for embedding, extracting, and evaluating the robustness of the watermark against common image processing attacks like JPEG compression.

## 📁 Project Structure

```text
sismul/
├── assets/         <-- Input directory
│   ├── host.jpg         # The original image to be watermarked
│   └── watermark.jpg    # The binary/logo image to embed
├── docs/           <-- Documentation
│   └── 18224062_Dokumentasi_Watermark.pdf  # Detailed project documentation
├── reports/        <-- Output directory
│   ├── watermarked_image.png    # Lossless result
│   ├── fidelity_report.png      # Visual quality analysis
│   ├── robustness_report.png    # JPEG attack analysis
│   ├── sample_success_qf[X].jpg # Success sample image
│   └── sample_fail_qf[Y].jpg    # Failed sample image
├── src/            <-- Source code
│   └── watermark.py     # Main script
├── requirements.txt
└── README.md
```

## 🚀 Features

- **DCT-Based Embedding:** Embeds watermark bits into the mid-frequency DCT coefficients of the Luminance (Y) channel.
- **Robustness Testing:** Automatically tests the watermark against JPEG compression from QF 100 to 10.
- **Automated Reporting:** Generates comprehensive visual reports and saves sample images of the results.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Joweeverse/sismul.git
   cd sismul
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

Run the script from the project root:

```bash
python src/watermark.py
```

### 📥 Inputs
Place your images in the `assets/` folder:
- **`host.jpg`**: Any color image.
- **`watermark.jpg`**: A small black-and-white or grayscale logo.

### 📤 Outputs
Results are generated in the `reports/` folder:
- **`watermarked_image.png`**: The final image containing the hidden watermark.
- **`fidelity_report.png`**: A comparison showing the "Invisibility" of the watermark (PSNR/SSIM).
- **`robustness_report.png`**: A graph showing how well the watermark survives JPEG compression.
- **`sample_success/fail.jpg`**: Visual proof of the image at different Quality Factors.

## 📊 Results Summary

- **Invisibility:** High PSNR ensures the watermark doesn't ruin the photo quality.
- **Robustness:** The script identifies the specific "Failure Point" where the watermark becomes unreadable due to compression.

## 📚 Documentation

Detailed technical documentation, including implementation details and experimental results, can be found here: [18224062_Dokumentasi_Watermark.pdf](docs/18224062_Dokumentasi_Watermark.pdf).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
