import os
from PIL import Image, ImageDraw

# Constants
DIGIT_WIDTH = 13  # New width as requested
DIGIT_HEIGHT = 23  # New height as requested
SEGMENT_WIDTH = 2  # Width of each segment (reduced for smaller display)
GAP = 1  # Gap between segments
BACKGROUND_COLOR = (0, 0, 0)  # Black background
SEGMENT_ON_COLOR = (255, 0, 0)  # Red LED color
SEGMENT_OFF_COLOR = (50, 0, 0)  # Dark red for off segments

# Define the segments for each digit (0-9)
# Segments are: [top, top-right, bottom-right, bottom, bottom-left, top-left, middle]
DIGIT_SEGMENTS = {
    0: [1, 1, 1, 1, 1, 1, 0],
    1: [0, 1, 1, 0, 0, 0, 0],
    2: [1, 1, 0, 1, 1, 0, 1],
    3: [1, 1, 1, 1, 0, 0, 1],
    4: [0, 1, 1, 0, 0, 1, 1],
    5: [1, 0, 1, 1, 0, 1, 1],
    6: [1, 0, 1, 1, 1, 1, 1],
    7: [1, 1, 1, 0, 0, 0, 0],
    8: [1, 1, 1, 1, 1, 1, 1],
    9: [1, 1, 1, 1, 0, 1, 1],
    # Add special characters
    10: [0, 0, 0, 0, 0, 0, 1],  # Minus sign
    11: [0, 0, 0, 0, 0, 0, 0],  # Blank/off
}

def draw_segment(draw, segments, segment_index, x, y, width, height, h_segment=True):
    """Draw a horizontal or vertical segment based on the segment index"""
    if segments[segment_index]:
        color = SEGMENT_ON_COLOR
    else:
        color = SEGMENT_OFF_COLOR
    
    if h_segment:  # Horizontal segment
        # Draw a trapezoid for better look
        points = [
            (x + GAP, y),
            (x + width - GAP, y),
            (x + width - GAP - SEGMENT_WIDTH//2, y + SEGMENT_WIDTH),
            (x + GAP + SEGMENT_WIDTH//2, y + SEGMENT_WIDTH)
        ]
        draw.polygon(points, fill=color)
    else:  # Vertical segment
        # Draw a trapezoid for better look
        points = [
            (x, y + GAP),
            (x + SEGMENT_WIDTH, y + GAP + SEGMENT_WIDTH//2),
            (x + SEGMENT_WIDTH, y + height - GAP - SEGMENT_WIDTH//2),
            (x, y + height - GAP)
        ]
        draw.polygon(points, fill=color)

def create_digit_image(digit):
    """Create an image for a 7-segment digit (0-9, -, blank)"""
    # Create a new image with black background
    img = Image.new('RGB', (DIGIT_WIDTH, DIGIT_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Get segments configuration for this digit
    if digit == '-':
        segments = DIGIT_SEGMENTS[10]  # Minus sign
    elif digit == ' ':
        segments = DIGIT_SEGMENTS[11]  # Blank
    else:
        segments = DIGIT_SEGMENTS[int(digit)]
    
    # Calculate positions
    h_segment_length = DIGIT_WIDTH - 2*GAP - SEGMENT_WIDTH
    v_segment_length = (DIGIT_HEIGHT - 3*GAP - SEGMENT_WIDTH) // 2
    
    # Draw top horizontal segment (0)
    draw_segment(draw, segments, 0, GAP, GAP, h_segment_length, SEGMENT_WIDTH)
    
    # Draw top-right vertical segment (1)
    draw_segment(draw, segments, 1, DIGIT_WIDTH - SEGMENT_WIDTH - GAP, 2*GAP + SEGMENT_WIDTH, 
                SEGMENT_WIDTH, v_segment_length, h_segment=False)
    
    # Draw bottom-right vertical segment (2)
    draw_segment(draw, segments, 2, DIGIT_WIDTH - SEGMENT_WIDTH - GAP, 
                DIGIT_HEIGHT - v_segment_length - 2*GAP, 
                SEGMENT_WIDTH, v_segment_length, h_segment=False)
    
    # Draw bottom horizontal segment (3)
    draw_segment(draw, segments, 3, GAP, DIGIT_HEIGHT - SEGMENT_WIDTH - GAP, 
                h_segment_length, SEGMENT_WIDTH)
    
    # Draw bottom-left vertical segment (4)
    draw_segment(draw, segments, 4, GAP, DIGIT_HEIGHT - v_segment_length - 2*GAP, 
                SEGMENT_WIDTH, v_segment_length, h_segment=False)
    
    # Draw top-left vertical segment (5)
    draw_segment(draw, segments, 5, GAP, 2*GAP + SEGMENT_WIDTH, 
                SEGMENT_WIDTH, v_segment_length, h_segment=False)
    
    # Draw middle horizontal segment (6)
    draw_segment(draw, segments, 6, GAP, DIGIT_HEIGHT//2 - SEGMENT_WIDTH//2, 
                h_segment_length, SEGMENT_WIDTH)
    
    return img

def generate_all_digit_images():
    """Generate images for all digits (0-9) and special characters"""
    # Create output directory if it doesn't exist
    os.makedirs('assets/digits', exist_ok=True)
    
    # Generate images for digits 0-9
    for digit in range(10):
        img = create_digit_image(str(digit))
        img.save(f'assets/digits/{digit}_digit.png')
        print(f"Generated digit image: {digit}_digit.png")
    
    # Generate minus sign
    img = create_digit_image('-')
    img.save('assets/digits/minus_digit.png')
    print("Generated digit image: minus_digit.png")
    
    # Generate blank/off digit
    img = create_digit_image(' ')
    img.save('assets/digits/blank_digit.png')
    print("Generated digit image: blank_digit.png")

if __name__ == "__main__":
    generate_all_digit_images()
    print("All digit images generated!")
