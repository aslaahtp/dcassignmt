from PIL import Image
import numpy as np

def jpeg_compression(image_path, quality=10):
    # Open the image
    img = Image.open(image_path)

    # Convert image to YCbCr color space
    img_yuv = img.convert('YCbCr')

    # Downsample chroma channels (Cb and Cr) using 4:2:0 chroma subsampling
    img_y, img_cb, img_cr = img_yuv.split()

    # Convert image data to numpy arrays
    y_data = np.array(img_y)
    cb_data = np.array(img_cb)
    cr_data = np.array(img_cr)

    # Apply DCT to 8x8 blocks of image data
    y_dct = np.block([[dct2(y_data[i:i+8, j:j+8]) for j in range(0, y_data.shape[1], 8)] for i in range(0, y_data.shape[0], 8)])
    cb_dct = np.block([[dct2(cb_data[i:i+8, j:j+8]) for j in range(0, cb_data.shape[1], 8)] for i in range(0, cb_data.shape[0], 8)])
    cr_dct = np.block([[dct2(cr_data[i:i+8, j:j+8]) for j in range(0, cr_data.shape[1], 8)] for i in range(0, cr_data.shape[0], 8)])

    # Quantize DCT coefficients
    y_quantized = quantize(y_dct , quality)
    cb_quantized = quantize(cb_dct, quality)
    cr_quantized = quantize(cr_dct, quality)

    # Reconstruct the image from quantized DCT coefficients
    y_reconstructed = np.block([[idct2(y_quantized[i:i+8, j:j+8]) for j in range(0, y_quantized.shape[1], 8)] for i in range(0, y_quantized.shape[0], 8)])
    cb_reconstructed = np.block([[idct2(cb_quantized[i:i+8, j:j+8]) for j in range(0, cb_quantized.shape[1], 8)] for i in range(0, cb_quantized.shape[0], 8)])
    cr_reconstructed = np.block([[idct2(cr_quantized[i:i+8, j:j+8]) for j in range(0, cr_quantized.shape[1], 8)] for i in range(0, cr_quantized.shape[0], 8)])

    # Merge Y, Cb, Cr channels
    reconstructed_img_yuv = Image.merge('YCbCr', (Image.fromarray(y_reconstructed.astype(np.uint8)), 
                                                  Image.fromarray(cb_reconstructed.astype(np.uint8)), 
                                                  Image.fromarray(cr_reconstructed.astype(np.uint8))))

    # Convert back to RGB color space
    compressed_img = reconstructed_img_yuv.convert('RGB')

    return compressed_img

# Quantization function
def quantize(block, quality):
    quantization_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                   [12, 12, 14, 19, 26, 58, 60, 55],
                                   [14, 13, 16, 24, 40, 57, 69, 56],
                                   [14, 17, 22, 29, 51, 87, 80, 62],
                                   [18, 22, 37, 56, 68, 109, 103, 77],
                                   [24, 35, 55, 64, 81, 104, 113, 92],
                                   [49, 64, 78, 87, 103, 121, 120, 101],
                                   [72, 92, 95, 98, 112, 100, 103, 99]])

    if quality < 50:
        scale = 50 / quality
    else:
        scale = 2 - (quality * 2) / 100
    
    # Split the block into 8x8 sub-blocks
    num_rows, num_cols = block.shape
    num_sub_blocks_row = num_rows // 8
    num_sub_blocks_col = num_cols // 8
    quantized_sub_blocks = []

    for i in range(num_sub_blocks_row):
        for j in range(num_sub_blocks_col):
            sub_block = block[i*8:(i+1)*8, j*8:(j+1)*8]
            quantized_sub_block = np.round(sub_block / (quantization_table * scale))
            quantized_sub_blocks.append(quantized_sub_block)

    # Combine the quantized sub-blocks into a single array
    quantized_block = np.block([[quantized_sub_blocks[j + i * num_sub_blocks_col] for j in range(num_sub_blocks_col)] for i in range(num_sub_blocks_row)])

    return quantized_block

# Discrete Cosine Transform
def dct2(block):
    return np.fft.fft2(block, norm="ortho")

# Inverse Discrete Cosine Transform
def idct2(block):
    return np.fft.ifft2(block, norm="ortho").real



# Example usage
input_image_path="sample.bmp"
# image_opened = Image.open(input_image_path)

# Display the image
# image.show()

output_image_path = "compressed_image.jpg"
compressed_img = jpeg_compression(input_image_path, quality=95)
compressed_img.save(output_image_path)
