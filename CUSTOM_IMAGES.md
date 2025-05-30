# Minesweeper Custom Cell Images

This version of Minesweeper supports custom image files for all game cells. The images are located in the `assets/` directory.

## Available Cell Images

The game uses the following PNG image files:

1. **hidden_cell.png** - The default appearance of an unrevealed cell
2. **empty_cell.png** - A revealed cell with no adjacent mines
3. **mine_cell.png** - A mine
4. **mine_red_cell.png** - A mine that was clicked (game over)
5. **flag_cell.png** - A flag marking a potential mine
6. **1_cell.png through 8_cell.png** - Revealed cells showing the number of adjacent mines (1-8)

## How to Customize

To customize the appearance of the game:

1. Create your own PNG image files with the same names as listed above
2. Place them in the `assets/` directory
3. Make sure the images are appropriately sized (recommended: 16x16 or 24x24 pixels)
4. Restart the game to see your custom images

## Image Requirements

- **File Format**: PNG is recommended for best quality and transparency support
- **Size**: Keep images square and relatively small (16x16 or 24x24 pixels work best)
- **Names**: The filenames must match exactly as listed above
- **Transparency**: You can use transparency in your PNG files for better visual effects

## Fallback Behavior

If any image file is missing, the game will display a text alternative:
- Numbers will be displayed with appropriate colors
- Mines will be displayed as bomb emojis (💣)
- Flags will be displayed as flag emojis (🚩)
