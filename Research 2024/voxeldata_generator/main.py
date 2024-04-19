from PIL import Image

def print_rgb_values(image_path):
    # Open the image file
    img = Image.open(image_path)
    string_writer = ""
    # Get the size of the image
    width, height = img.size

    # Iterate through each pixel and print RGB values
    for y in range(height):
        for x in range(width):
            # Get RGB values of the pixel at (x, y)
            pixel = img.getpixel((x, y))

            # Print RGB values
            string_writer += str(pixel[0]) + "\n"
            # print(f"Pixel at ({x}, {y}): R={pixel[0]}, G={pixel[1]}, B={pixel[2]}")

    return string_writer

if __name__ == "__main__":
    # Replace 'image.png' with the path to your PNG image
    f = open("voxeldata.txt", "w")
    image_path = '2newshepplogan.png'
    voxel_str = (print_rgb_values(image_path))
    """
    strr = ""
    for i in range(256):
        strr += (voxel_str)
    for i in range(512 * 512 * 255 - 1):
        strr += ("0\n")
    strr += "0"""
    f.write(voxel_str)
    print("g")
    f.close()
    f.close()
