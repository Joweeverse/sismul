import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_images(host_path, watermark_path):
    """
    Loads the host and watermark images.
    The watermark is converted to grayscale and binarized.
    """
    if not os.path.exists(host_path) or not os.path.exists(watermark_path): #check if files exist
        raise FileNotFoundError(f"Ensure {host_path} and {watermark_path} exist.")
    
    host = cv2.imread(host_path) # load in color (BGR)
    if host is None:
        raise ValueError(f"Could not load host image: {host_path}")
        
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE) # load the watermarkin grayscale (black or white)
    if watermark is None:
        raise ValueError(f"Could not load watermark image: {watermark_path}")
        
    # Binarize the watermark (0 or 255)
    _, watermark_bin = cv2.threshold(watermark, 127, 255, cv2.THRESH_BINARY)
    
    return host, watermark_bin

def prepare_watermark(watermark, host_shape, block_size=8): 
    """
    Resizes the watermark to fit the grid of blocks in the host image.
    If host is HxW, watermark will be (H/block_size) x (W/block_size).
    """
    h, w = host_shape[:2] # height and width of the host image
    rows = h // block_size # number of blocks vertically
    cols = w // block_size # number of blocks horizontally
    
    # Resize using nearest neighbor to preserve binary nature
    resized_wm = cv2.resize(watermark, (cols, rows), interpolation=cv2.INTER_NEAREST) # resize the watermark to fit the number of blocks in the host image
    return (resized_wm > 127).astype(np.uint8) # convert to binary (0 and 1)

def embed_watermark(host, watermark, block_size=8, alpha=15):
    """
    Embeds binary watermark bits into the DCT coefficients of the host image's Y channel.
    The embedding modifies the (4,4), a mid-frequency coefficient to balance a good image quality and robustness.
    """
    # 1. Color space conversion (BGR -> YCrCb) to process Luminance (Y)
    ycrcb = cv2.cvtColor(host, cv2.COLOR_BGR2YCrCb) # manipulate the luminance channel for better invisibility
    y_channel = ycrcb[:, :, 0].astype(np.float32) # convert to float for DCT processing
    
    wm_h, wm_w = watermark.shape
    
    # Mid-frequency coefficient choice (4,4) balances robustness and invisibility
    coeff_pos = (4, 4) # The common choice for mid-frequency embedding, which offers a good balance between invisibility and robustness against JPEG compression
    
    # 2. Block-based DCT Embedding
    for i in range(wm_h):
        for j in range(wm_w):
            row_start = i * block_size
            col_start = j * block_size
            
            # Extract 8x8 block
            block = y_channel[row_start:row_start+block_size, col_start:col_start+block_size]
            
            # Apply Discrete Cosine Transform
            dct_block = cv2.dct(block)
            
            # Embed bit: Additive modification
            if watermark[i, j] == 1:
                dct_block[coeff_pos] += alpha # Increase the coefficient to embed a '1' bit
            else:
                dct_block[coeff_pos] -= alpha
                
            # Apply Inverse DCT
            y_channel[row_start:row_start+block_size, col_start:col_start+block_size] = cv2.idct(dct_block)
            
    # 3. Reconstruct image
    y_channel = np.clip(y_channel, 0, 255).astype(np.uint8)
    ycrcb[:, :, 0] = y_channel
    watermarked_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return watermarked_img

def extract_watermark(watermarked_img, host_img, block_size=8):
    """
    Extracts the watermark by comparing DCT coefficients of the watermarked image 
    against the original host image. This non-blind method is highly robust 
    against JPEG compression.
    
    Returns:
        A reconstructed binary watermark image.
    """
    # 1. Process the watermarked image (likely the compressed version)
    ycrcb_wm = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YCrCb)
    y_wm = ycrcb_wm[:, :, 0].astype(np.float32)
    
    # 2. Process the original host for comparison
    ycrcb_orig = cv2.cvtColor(host_img, cv2.COLOR_BGR2YCrCb)
    y_orig = ycrcb_orig[:, :, 0].astype(np.float32)
    
    h, w = y_wm.shape
    rows, cols = h // block_size, w // block_size
    extracted_wm = np.zeros((rows, cols), dtype=np.uint8)
    
    # Synchronized coefficient position (must match embedding)
    coeff_pos = (4, 4)
    
    # 3. Block-based Extraction
    for i in range(rows):
        for j in range(cols):
            row_s, col_s = i * block_size, j * block_size
            
            # Extract and transform blocks from both images
            block_wm = cv2.dct(y_wm[row_s:row_s+block_size, col_s:col_s+block_size])
            block_orig = cv2.dct(y_orig[row_s:row_s+block_size, col_s:col_s+block_size])
            
            # Logic: If the watermarked coefficient is greater than the original, 
            # it indicates a bit '1' was embedded (assuming alpha > 0).
            if block_wm[coeff_pos] > block_orig[coeff_pos]:
                extracted_wm[i, j] = 255  # White pixel
            else:
                extracted_wm[i, j] = 0    # Black pixel
                
    return extracted_wm

def create_dummy_images():
    """Generates sample 'host.jpg' and 'watermark.jpg' for demonstration."""
    print("Generating dummy images for testing...")
    
    # Create a 512x512 color gradient (host)
    host = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        host[i, :, 0] = i // 2  # Blue channel gradient
        host[:, i, 1] = i // 2  # Green channel gradient
    cv2.imwrite('host.jpg', host)
    
    # Create a simple 64x64 binary image (watermark)
    wm = np.zeros((64, 64), dtype=np.uint8)
    cv2.putText(wm, "COPY", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    cv2.imwrite('watermark.jpg', wm)

def calculate_ber(original_wm, extracted_wm):
    """
    Calculates the Bit Error Rate (BER) between the original and extracted watermark.
    BER = (Number of mismatched bits) / (Total number of bits)
    """
    # Ensure they are the same shape
    if original_wm.shape != extracted_wm.shape:
        extracted_wm = cv2.resize(extracted_wm, (original_wm.shape[1], original_wm.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Binarize to 0 and 1 for bitwise comparison. 
    # Use > 0 to handle both 0/1 and 0/255 formats.
    orig_bits = (original_wm > 0).astype(np.uint8)
    ext_bits = (extracted_wm > 0).astype(np.uint8)
    
    mismatches = np.sum(orig_bits != ext_bits)
    total_bits = orig_bits.size
    
    return mismatches / total_bits

def calculate_psnr(img1, img2):
    """Calculates PSNR between two images manually."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculates a simplified SSIM using OpenCV's structural similarity logic."""
    # Convert to grayscale for structural analysis
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Constants to stabilize the division with weak denominator
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = gray1.astype(np.float32)
    img2 = gray2.astype(np.float32)
    kernel = np.ones((11, 11), np.float32) / 121

    mu1 = cv2.filter2D(img1, -1, kernel)
    mu2 = cv2.filter2D(img2, -1, kernel)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, kernel) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, kernel) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def evaluate_and_visualize(watermarked_img, host_img, original_wm_bits, report_dir, save_path='robustness_report.png', qf_range=range(100, 0, -10)):
    """
    Core Assignment Task: Evaluate watermark robustness against JPEG compression.
    Identifies the Quality Factor (QF) threshold where extraction fails.
    Also saves two sample images: one successful and one failed.
    """
    qfs = []
    bers = []        # Bit Error Rate
    psnr_values = [] # Host Image Quality
    
    failure_threshold_ber = 0.2 
    qf_failure_point = None
    
    # Track samples to save
    success_sample = None # (qf, image)
    fail_sample = None    # (qf, image)

    print(f"\n{'JPEG Quality (QF)':<20} | {'Watermark Error (BER)':<25} | {'Status'}")
    print("-" * 70)

    for qf in qf_range:
        # 1. Apply JPEG Compression Attack
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qf]
        _, encoded_img = cv2.imencode('.jpg', watermarked_img, encode_param)
        compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        # 2. Extract Watermark
        extracted_wm = extract_watermark(compressed_img, host_img)
        ber = calculate_ber(original_wm_bits, extracted_wm)
        
        # 3. Record Data
        qfs.append(qf)
        bers.append(ber)
        psnr_values.append(calculate_psnr(host_img, compressed_img))
        
        # 4. Determine status and pick samples
        status = "RECOVERABLE"
        if ber > failure_threshold_ber:
            if qf_failure_point is None:
                qf_failure_point = qf
                status = "!!! FAILED !!!"
                fail_sample = (qf, compressed_img)
            else:
                status = "FAILED"
        else:
            # Keep the lowest QF that still succeeds as our success sample
            success_sample = (qf, compressed_img)
            
        print(f"{qf:<20} | {ber:<25.4f} | {status}")

    # --- SAVE SAMPLE IMAGES ---
    if success_sample:
        qf, img = success_sample
        path = os.path.join(report_dir, f'sample_success_qf{qf}.jpg')
        cv2.imwrite(path, img)
        print(f"[*] Saved successful sample (QF {qf}) to: {os.path.basename(path)}")
        
    if fail_sample:
        qf, img = fail_sample
        path = os.path.join(report_dir, f'sample_fail_qf{qf}.jpg')
        cv2.imwrite(path, img)
        print(f"[*] Saved failed sample (QF {qf}) to:     {os.path.basename(path)}")

    # --- FINAL VISUALIZATION ---
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: BER vs Quality Factor (The most important graph for your task)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(qfs, bers, 'r-o', linewidth=2, label='Bit Error Rate (BER)')
    ax1.axhline(y=failure_threshold_ber, color='blue', linestyle='--', label='Failure Limit')
    
    if qf_failure_point:
        ax1.annotate(f'WATERMARK FAILS AT QF={qf_failure_point}', 
                     xy=(qf_failure_point, failure_threshold_ber), 
                     xytext=(qf_failure_point+15, failure_threshold_ber+0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=12, fontweight='bold', color='red')

    ax1.set_title('Robustness Evaluation: At what QF does the watermark fail?', fontsize=14)
    ax1.set_xlabel('JPEG Quality Factor (Higher = Less Compression)')
    ax1.set_ylabel('Watermark Error (BER)')
    ax1.invert_xaxis() # Show 100 -> 10
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Visual proof of extracted watermarks
    # Show the watermark images side-by-side to visually show degradation
    sample_qfs = [100, 70, 40, 20, 10]
    for i, qf in enumerate(sample_qfs):
        # Extract for this specific QF
        _, enc = cv2.imencode('.jpg', watermarked_img, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        ext = extract_watermark(dec, host_img)
        
        ax = fig.add_subplot(2, 5, 6 + i)
        ax.imshow(ext, cmap='gray')
        ax.set_title(f'Extracted @ QF={qf}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    
    print(f"\n[+] ANALYSIS COMPLETE")
    if qf_failure_point:
        print(f"[!] The watermark becomes unextractable at QF = {qf_failure_point}")
    else:
        print("[+] The watermark survived all compression levels (QF 100 to 10)!")
    print(f"[+] Final report saved as '{save_path}'")
    plt.show()

def generate_fidelity_report(original_img, watermarked_img, save_path='fidelity_report.png'):
    """
    Creates a dedicated report comparing the original photo vs the watermarked photo.
    This shows the "Invisibility" performance of the watermark.
    """
    print("[*] Generating Image Fidelity Report...")
    
    # 1. Calculate Metrics
    current_psnr = calculate_psnr(original_img, watermarked_img)
    current_ssim = calculate_ssim(original_img, watermarked_img)
    
    # 2. Calculate Difference Map (multiplied by 10 to make changes visible to the eye)
    # This shows exactly WHICH pixels were changed by the DCT embedding
    diff = cv2.absdiff(original_img, watermarked_img)
    diff_visible = diff * 10 
    
    # 3. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    # Original Photo
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Original Photo", fontsize=12)
    axes[0].axis('off')
    
    # Watermarked Photo
    axes[1].imshow(cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"2. Watermarked Photo\nPSNR: {current_psnr:.2f} dB", fontsize=12)
    axes[1].axis('off')
    
    # Difference Map
    axes[2].imshow(cv2.cvtColor(diff_visible, cv2.COLOR_BGR2RGB))
    axes[2].set_title("3. Where pixels changed?\n(Difference Map x10 for visibility)", fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle("IMAGE FIDELITY ANALYSIS: Does the watermark ruin the photo?", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(save_path)
    print(f"[+] Fidelity report saved as '{save_path}'")
    
def main():
    """
    Main execution flow:
    1. Setup directories and paths
    2. Load host and watermark images
    3. Process and embed watermark
    4. Save results and generate analysis reports
    """
    # --- 1. SETUP DIRECTORIES ---
    # Determine the project structure relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Input paths (from sismul/assets)
    HOST_IMG_PATH = os.path.join(project_root, 'assets', 'host.jpg')
    WATERMARK_IMG_PATH = os.path.join(project_root, 'assets', 'watermark.jpg')
    
    # Output paths (to sismul/reports)
    REPORT_DIR = os.path.join(project_root, 'reports')
    if not os.path.exists(REPORT_DIR):
        print(f"[*] Creating output directory: {REPORT_DIR}")
        os.makedirs(REPORT_DIR)
        
    OUTPUT_IMG = os.path.join(REPORT_DIR, 'watermarked_image.png')
    FIDELITY_REPORT = os.path.join(REPORT_DIR, 'fidelity_report.png')
    ROBUSTNESS_REPORT = os.path.join(REPORT_DIR, 'robustness_report.png')
    
    print("="*60)
    print(" DIGITAL WATERMARKING SYSTEM - DCT DOMAIN ")
    print("="*60)
    print(f"Project Root: {project_root}")
    print(f"Input Host:   {os.path.basename(HOST_IMG_PATH)}")
    print(f"Input Mark:   {os.path.basename(WATERMARK_IMG_PATH)}")
    print(f"Output Dir:   {os.path.basename(REPORT_DIR)}/")
    print("-" * 60)

    # --- 2. VALIDATION ---
    if not os.path.exists(HOST_IMG_PATH) or not os.path.exists(WATERMARK_IMG_PATH):
        print(f"[!] ERROR: Missing input assets.")
        print(f"    Please ensure the following files exist:")
        print(f"    - {HOST_IMG_PATH}")
        print(f"    - {WATERMARK_IMG_PATH}")
        return
    
    try:
        # --- 3. LOADING & PREPARATION ---
        print(f"[*] Step 1: Loading images...")
        host, watermark = load_images(HOST_IMG_PATH, WATERMARK_IMG_PATH)
        
        print("[*] Step 2: Preparing binary watermark grid...")
        wm_binary = prepare_watermark(watermark, host.shape)
        
        # --- 4. EMBEDDING ---
        print("[*] Step 3: Embedding watermark using DCT (Alpha=30)...")
        watermarked_img = embed_watermark(host, wm_binary, alpha=30)
        
        # --- 5. SAVING & REPORTING ---
        print(f"[*] Step 4: Saving watermarked image to:\n    -> {OUTPUT_IMG}")
        cv2.imwrite(OUTPUT_IMG, watermarked_img)
        
        print(f"[*] Step 5: Generating Image Fidelity Report (Visual Quality)...")
        generate_fidelity_report(host, watermarked_img, FIDELITY_REPORT)
        
        print("\n[*] Step 6: Starting Robustness Evaluation (JPEG Compression Test)...")
        evaluate_and_visualize(watermarked_img, host, wm_binary, REPORT_DIR, ROBUSTNESS_REPORT)
        
        print("\n" + "="*60)
        print(" SUCCESS: All tasks completed successfully.")
        print(f" Results are located in: {REPORT_DIR}")
        print("="*60)
        
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
