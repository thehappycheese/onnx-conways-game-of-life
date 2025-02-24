import "./App.css";
import { createSignal, onCleanup, onMount } from "solid-js";
import { InferenceSession, Tensor } from "onnxruntime-web/webgpu";

const GRID_SIZE = 500;

// Gosper Glider Gun pattern
const gosperGliderGun = [
  [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  ],
  [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  ],
  [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
  ],
  [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
  ],
  [
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  ],
  [
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  ],
  [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  ],
  [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  ],
  [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  ],
];

function App() {
  let canvasRef;
  const [grid, setGrid] = createSignal(
    new Float32Array(1 * 1 * GRID_SIZE * GRID_SIZE),
  );

  const runSimulation = async () => {
    const session = await InferenceSession.create("./conway_game_of_life.onnx");

    // const initialGrid = new Float32Array(1 * 1 * GRID_SIZE * GRID_SIZE);
    // for (let i = 0; i < initialGrid.length; i++) {
    //   initialGrid[i] = Math.random() > 0.5 ? 1 : 0;
    // }
    // setGrid(initialGrid);

    // const initialGrid = new Float32Array(1 * 1 * GRID_SIZE * GRID_SIZE);
    // const centerX = Math.floor(GRID_SIZE / 2);
    // const centerY = Math.floor(GRID_SIZE / 2);

    // const setCell = (x, y, value) => {
    //     initialGrid[y * GRID_SIZE + x] = value;
    // };

    // // R-pentomino pattern (placed at the center)
    // setCell(centerX, centerY, 1);
    // setCell(centerX + 1, centerY, 1);
    // setCell(centerX - 1, centerY + 1, 1);
    // setCell(centerX, centerY + 1, 1);
    // setCell(centerX, centerY + 2, 1);

    // setGrid(initialGrid);

    // const initialGrid = new Float32Array(1 * 1 * GRID_SIZE * GRID_SIZE);

    // // Set the Gosper Glider Gun pattern in the center of the grid
    // const offsetX = Math.floor((GRID_SIZE - gosperGliderGun[0].length) / 2);
    // const offsetY = Math.floor((GRID_SIZE - gosperGliderGun.length) / 2);
    // for (let y = 0; y < gosperGliderGun.length; y++) {
    //   for (let x = 0; x < gosperGliderGun[y].length; x++) {
    //     initialGrid[(offsetY + y) * GRID_SIZE + (offsetX + x)] = gosperGliderGun[y][x];
    //   }
    // }

    // setGrid(initialGrid);

    const parsePattern = (patternString) => {
      const lines = patternString.trim().split("\n");
      const pattern = lines
        .filter((line) => line.trim() && !line.startsWith("!"))
        .map((line) => line.trim().split(""));

      return pattern;
    };

    const initializeGrid = (pattern) => {
      const initialGrid = new Float32Array(1 * 1 * GRID_SIZE * GRID_SIZE);

      const patternWidth = pattern[0].length;
      const patternHeight = pattern.length;
      const offsetX = Math.floor((GRID_SIZE - patternWidth) / 2);
      const offsetY = Math.floor((GRID_SIZE - patternHeight) / 2);

      for (let y = 0; y < patternHeight; y++) {
        for (let x = 0; x < patternWidth; x++) {
          const cell = pattern[y][x];
          if (cell === "O") {
            initialGrid[(offsetY + y) * GRID_SIZE + (offsetX + x)] = 1;
          }
        }
      }

      return initialGrid;
    };

    const patternString = `
        ! 52514m.cells
! https://conwaylife.com/wiki/52513M
! https://www.conwaylife.com/patterns/52514m.cells
.O.OO..OOOOO.OO.
OO...OOO..O...OO
..O.O.......O..O
.....O..O.OOO..O
OOO...O..OO...OO
....OOO.O...OOO.
....O..OOO.....O
...O.OO.O.....O.
O..OOO....OO.OOO
O.O.O..O..O.....
O......O..OO....
.OO.O..O....O..O
.O..OO....O.O.OO
..O.....OO...O..
O.OO.OOO....O..O
OO..............
    `;

    const pattern = parsePattern(patternString);
    const initialGrid = initializeGrid(pattern);
    setGrid(initialGrid);

    const simulate = async () => {
      const input = new Tensor("float32", grid(), [1, 1, GRID_SIZE, GRID_SIZE]);
      const output = await session.run({ input });
      const outputData = output.output.data;
      setGrid(new Float32Array(outputData));
      renderGrid(); // Call renderGrid after updating the grid state
      requestAnimationFrame(simulate);
    };

    simulate();
  };

  const renderGrid = () => {
    const canvas = canvasRef;
    if (!canvas || !grid()) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(GRID_SIZE, GRID_SIZE);
    const data = imageData.data;
    for (let i = 0; i < grid().length; i++) {
      const value = grid()[i] * 255;
      data[i * 4] = value;
      data[i * 4 + 1] = value;
      data[i * 4 + 2] = value;
      data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  };

  onMount(() => {
    runSimulation();
  });

  onCleanup(() => {
    // Cleanup code, if needed
  });

  return (
    <div class="App">
        <p>Browser GPU runtime being used to run ONNX implementation of conways game of life! 😁</p>
      <canvas
        ref={canvasRef}
        width={GRID_SIZE}
        height={GRID_SIZE}
        style={{
            width:(GRID_SIZE*2)+"px",
            height:(GRID_SIZE*2)+"px",
            "image-rendering": "pixelated",
        }}
      />
      <p>
        Although ONNX op codes are enough to implement the core logic, they do not support looping and streaming output. 
        We still have to use javascript to feed the output tensor back in as input every frame.
        This ends up using a lot of CPU, but still its kinda cool.
      </p>
    </div>
  );
}

export default App;
