from PIL import Image
import matplotlib.pyplot as plt


def append_images(images, direction='horizontal',
                  bg_color=(255, 255, 255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im


def show_pairs_on_images(filename1, filename2, pairs, hex_alpha='33'):
    img1 = Image.open(filename1)
    img2 = Image.open(filename2)

    images = [img1, img2]
    result_image = append_images(images, aligment='top')

    width, height = img1.size
    x_offset = width

    for pair in pairs:
        plt.plot([pair[0].pt[0], pair[1].pt[0] + x_offset], [pair[0].pt[1], pair[1].pt[1]], '#FFFF00' + hex_alpha)
        # plt.plot([pair[0].pt[0], pair[1].pt[0] + x_offset], [pair[0].pt[1], pair[1].pt[1]])

    plt.imshow(result_image)
    plt.show()
