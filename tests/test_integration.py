"""
Integration tests for the complete minesweeper game
Tests interaction between different components
"""

import pytest
from game.board import GameBoard, GameState, CellState


class TestGameIntegration:
    """Integration tests for complete game scenarios"""
    
    def test_complete_beginner_game_scenario(self):
        """Test a complete game scenario from start to finish"""
        board = GameBoard(9, 9, 10)  # Beginner
        
        # Game should start in READY state
        assert board.game_state == GameState.READY
        assert board.mines_placed is False
        
        # First click should start the game
        board.reveal_cell(4, 4)  # Click center
        assert board.game_state == GameState.PLAYING
        assert board.mines_placed is True
        assert board.cells_revealed >= 1
        
        # First clicked cell should not be a mine
        first_cell = board.get_cell(4, 4)
        assert first_cell.is_mine is False
        assert first_cell.state == CellState.REVEALED
        
        # Flag some cells
        board.toggle_flag(0, 0)
        board.toggle_flag(0, 1)
        assert board.flags_used == 2
        assert board.get_remaining_mines() == 8
    
    def test_losing_game_scenario(self):
        """Test a game where player hits a mine"""
        board = GameBoard(3, 3, 8)  # Almost all mines
        
        # Start game
        board.reveal_cell(1, 1)
        
        # Try to find a mine and click it
        mine_found = False
        for row in range(3):
            for col in range(3):
                cell = board.get_cell(row, col)
                if cell.is_mine and cell.state == CellState.HIDDEN:
                    board.reveal_cell(row, col)
                    mine_found = True
                    break
            if mine_found:
                break
        
        if mine_found:
            assert board.game_state == GameState.LOST
    
    def test_winning_game_scenario(self):
        """Test a game where player wins"""
        board = GameBoard(3, 3, 1)  # Only one mine
        
        # Manually place mine for predictable test
        board.board[0][0].place_mine()
        board._calculate_adjacent_mines()
        board.mines_placed = True
        board.game_state = GameState.PLAYING
        
        # Reveal all safe cells
        safe_cells = [
            (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2)
        ]
        
        for row, col in safe_cells:
            board.reveal_cell(row, col)
        
        assert board.game_state == GameState.WON
        # Mine should be automatically flagged
        assert board.get_cell(0, 0).state == CellState.FLAGGED
    def test_game_state_transitions(self):
        """Test all possible game state transitions"""
        board = GameBoard(3, 3, 1)
        
        # READY -> PLAYING
        assert board.game_state == GameState.READY
        board.reveal_cell(1, 1)
        assert board.game_state == GameState.PLAYING
        
        # Manually place mine in a specific position to make test deterministic
        # Clear any existing mine placement
        for row in range(3):
            for col in range(3):
                board.board[row][col].is_mine = False
        
        # Place mine at (0,0) - away from our test cells
        board.board[0][0].place_mine()
        board._calculate_adjacent_mines()
        
        # Test that game states are persistent
        board.toggle_flag(0, 1)  # Flag a different cell since (0,0) has mine
        assert board.game_state == GameState.PLAYING
        
        board.reveal_cell(2, 2)  # This cell should be safe and not trigger auto-win
        assert board.game_state == GameState.PLAYING

    def test_flag_counter_accuracy(self):
        """Test that flag counter stays accurate throughout game"""
        board = GameBoard(4, 4, 4)
        board.game_state = GameState.PLAYING
        
        # Flag several cells
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for i, (row, col) in enumerate(positions):
            board.toggle_flag(row, col)
            assert board.flags_used == i + 1
            assert board.get_remaining_mines() == 4 - (i + 1)
        
        # Unflag some cells
        board.toggle_flag(0, 0)
        assert board.flags_used == 3
        assert board.get_remaining_mines() == 1
        
        board.toggle_flag(1, 1)
        assert board.flags_used == 2
        assert board.get_remaining_mines() == 2
    
    def test_auto_reveal_chain_reaction(self):
        """Test that auto-reveal creates proper chain reactions"""
        # Create board with mines in corners only
        board = GameBoard(5, 5, 4)
        
        # Place mines in corners manually
        corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
        for row, col in corners:
            board.board[row][col].place_mine()
        
        board._calculate_adjacent_mines()
        board.mines_placed = True
        board.game_state = GameState.PLAYING
        
        # Click center cell (should auto-reveal many cells)
        board.reveal_cell(2, 2)
        
        # Should have revealed many cells due to chain reaction
        assert board.cells_revealed > 10  # Most of the center area
        
        # Center cell should have 0 adjacent mines
        center_cell = board.get_cell(2, 2)
        assert center_cell.adjacent_mines == 0
        assert center_cell.state == CellState.REVEALED
    
    def test_game_reset_preserves_board_size(self):
        """Test that game reset maintains board dimensions"""
        original_board = GameBoard(7, 11, 15)  # Custom size
        
        # Play some moves
        original_board.reveal_cell(3, 5)
        original_board.toggle_flag(0, 0)
        
        # Reset game
        original_board.reset_game()
        
        # Verify board size is preserved
        assert original_board.rows == 7
        assert original_board.cols == 11
        assert original_board.total_mines == 15
        
        # Verify all cells are reset
        for row in range(7):
            for col in range(11):
                cell = original_board.get_cell(row, col)
                assert cell.state == CellState.HIDDEN
                assert cell.is_mine is False
                assert cell.adjacent_mines == 0
    
    def test_difficulty_level_characteristics(self):
        """Test that each difficulty level has correct characteristics"""
        difficulties = {
            'beginner': (9, 9, 10),
            'intermediate': (16, 16, 40),
            'expert': (16, 30, 99)
        }
        
        for difficulty, (rows, cols, mines) in difficulties.items():
            board = GameBoard()
            board.reset_game(difficulty)
            
            assert board.rows == rows
            assert board.cols == cols
            assert board.total_mines == mines
            
            # Test mine density is reasonable
            total_cells = rows * cols
            mine_density = mines / total_cells
            assert 0.1 <= mine_density <= 0.25  # 10-25% mine density


class TestGameStatePersistence:
    """Test that game state is properly maintained"""
    
    def test_revealed_cells_count_accuracy(self):
        """Test that revealed cells count is accurate"""
        board = GameBoard(4, 4, 2)
        
        # Manually set up predictable board
        board.board[0][0].place_mine()
        board.board[3][3].place_mine()
        board._calculate_adjacent_mines()
        board.mines_placed = True
        board.game_state = GameState.PLAYING
        
        # Reveal specific cells
        reveal_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (row, col) in enumerate(reveal_positions):
            board.reveal_cell(row, col)
            assert board.cells_revealed >= i + 1  # May auto-reveal more
        
        # Count actual revealed cells
        actual_revealed = 0
        for row in range(4):
            for col in range(4):
                if board.get_cell(row, col).state == CellState.REVEALED:
                    actual_revealed += 1
        
        assert board.cells_revealed == actual_revealed
    
    def test_mine_placement_randomness(self):
        """Test that mine placement is properly randomized"""
        # Create multiple boards and check mine distributions
        mine_positions = set()
        
        for _ in range(20):  # Create 20 different boards
            board = GameBoard(4, 4, 4)
            board.reveal_cell(0, 0)  # Trigger mine placement
            
            # Collect mine positions
            board_mines = []
            for row in range(4):
                for col in range(4):
                    if board.get_cell(row, col).is_mine:
                        board_mines.append((row, col))
            
            mine_positions.add(tuple(sorted(board_mines)))
        
        # Should have multiple different mine configurations
        assert len(mine_positions) > 1
    
    def test_adjacent_mine_counts_correctness(self):
        """Test that adjacent mine counts are calculated correctly"""
        board = GameBoard(3, 3, 0)  # Start with no mines
        
        # Place mines in specific pattern
        # X . X
        # . . .
        # X . X
        mine_positions = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for row, col in mine_positions:
            board.board[row][col].place_mine()
        
        board._calculate_adjacent_mines()
        
        # Check expected adjacent counts
        expected_counts = {
            (0, 1): 2,  # Between two mines
            (1, 0): 2,  # Between two mines
            (1, 1): 4,  # Center - adjacent to all 4 mines
            (1, 2): 2,  # Between two mines
            (2, 1): 2,  # Between two mines
        }
        
        for (row, col), expected in expected_counts.items():
            actual = board.get_cell(row, col).adjacent_mines
            assert actual == expected, f"Cell ({row},{col}) expected {expected}, got {actual}"
