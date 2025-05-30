"""
Unit tests for the Cell class
Tests individual cell behavior and state management
"""

import pytest
from game.board import Cell, CellState


class TestCell:
    """Test cases for the Cell class"""
    
    def test_cell_initialization(self):
        """Test that cell initializes with correct default values"""
        cell = Cell(3, 5)
        
        assert cell.row == 3
        assert cell.col == 5
        assert cell.is_mine is False
        assert cell.state == CellState.HIDDEN
        assert cell.adjacent_mines == 0
    
    def test_place_mine(self):
        """Test placing a mine in a cell"""
        cell = Cell(0, 0)
        assert cell.is_mine is False
        
        cell.place_mine()
        assert cell.is_mine is True
    
    def test_reveal_hidden_cell(self):
        """Test revealing a hidden cell returns True"""
        cell = Cell(0, 0)
        assert cell.state == CellState.HIDDEN
        
        result = cell.reveal()
        assert result is True
        assert cell.state == CellState.REVEALED
    
    def test_reveal_already_revealed_cell(self):
        """Test revealing an already revealed cell returns False"""
        cell = Cell(0, 0)
        cell.reveal()  # First reveal
        
        result = cell.reveal()  # Second reveal
        assert result is False
        assert cell.state == CellState.REVEALED
    
    def test_reveal_flagged_cell(self):
        """Test revealing a flagged cell returns False"""
        cell = Cell(0, 0)
        cell.toggle_flag()
        assert cell.state == CellState.FLAGGED
        
        result = cell.reveal()
        assert result is False
        assert cell.state == CellState.FLAGGED
    
    def test_toggle_flag_from_hidden(self):
        """Test flagging a hidden cell"""
        cell = Cell(0, 0)
        assert cell.state == CellState.HIDDEN
        
        cell.toggle_flag()
        assert cell.state == CellState.FLAGGED
    
    def test_toggle_flag_from_flagged(self):
        """Test unflagging a flagged cell"""
        cell = Cell(0, 0)
        cell.toggle_flag()  # Flag it
        assert cell.state == CellState.FLAGGED
        
        cell.toggle_flag()  # Unflag it
        assert cell.state == CellState.HIDDEN
    
    def test_toggle_flag_on_revealed_cell(self):
        """Test that revealed cells cannot be flagged"""
        cell = Cell(0, 0)
        cell.reveal()
        assert cell.state == CellState.REVEALED
        
        cell.toggle_flag()
        assert cell.state == CellState.REVEALED  # Should remain revealed
    
    def test_is_revealed(self):
        """Test is_revealed method"""
        cell = Cell(0, 0)
        assert cell.is_revealed() is False
        
        cell.reveal()
        assert cell.is_revealed() is True
    
    def test_is_flagged(self):
        """Test is_flagged method"""
        cell = Cell(0, 0)
        assert cell.is_flagged() is False
        
        cell.toggle_flag()
        assert cell.is_flagged() is True
        
        cell.toggle_flag()
        assert cell.is_flagged() is False
