# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GitHub Pages static site (`MIKEK8.github.io`) hosting interactive HTML5 demonstrations. It contains four standalone applications with no build process, no dependencies to install, and no backend.

## Development

**No build/test/lint commands exist.** Files are served directly by GitHub Pages.

To develop locally, simply open any HTML file in a browser. For laberint.html, you can use the `?size=N` query parameter to control maze size (must be odd, minimum 5, defaults to 41).

## Architecture

### Applications

1. **index.html** - Landing page with links to all demos

2. **neural-xor.html** - Neural network trainer that learns XOR logic gate
   - **Goal**: Find the ideal training approach that achieves 100% convergence rate for any random initial weights
   - Architecture: Input(2) → Hidden(1) + skip connections → Output(1), total 7 weights
   - Uses batch backpropagation with sigmoid activation
   - Adaptive learning rate with oscillation detection
   - Vue.js 2 for UI, Google Charts API (LineChart) for visualization
   - CLI version: `neural-xor-cli.js` for testing convergence rates
   - Current params: startLR=5, minLR=0.5, maxLR=20, maxIterations=5000

3. **laberint.html** - Single-path maze solver
   - Generates mazes using randomized Depth-First Search
   - Solves using DFS with animated path visualization
   - HTML5 Canvas rendering with requestAnimationFrame

4. **laberint-2.html** - Multi-path maze solver
   - Concurrent pathfinding (2 paths: A→D and B→C)
   - Uses async/await for simultaneous animations
   - Different colors per path (#e74c3c, #f1c40f)

5. **midi.html** - Web Audio API synthesizer with 25 note buttons

### External Dependencies (CDN only)

- Google Charts API (gstatic.com)
- Vue.js 2 (cdn.jsdelivr.net)

## Key Implementation Details

- Chart redraw uses 200ms debouncing to prevent rendering conflicts
- Maze animation: 8 line-draw steps per cell, 36ms per cell for multi-path
- Neural network convergence: all outputs must have < 0.1 error
- laberint.html and laberint-2.html share similar structure and could benefit from shared utilities
