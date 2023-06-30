import React, { useState, useEffect } from "react";

import { Board } from "./components/Board";
import { ResetButton } from "./components/ResetButton";
import { ScoreBoard } from "./components/ScoreBoard";

import { useQuery } from "react-query";
import axios from "axios";

import './App.css';

const App = () => {

  const WIN_CONDITIONS = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6]
  ]

  const [xPlaying, setXPlaying] = useState(true);
  const [board, setBoard] = useState(Array(9).fill(null));
  const [scores, setScores] = useState({ xScore: 0, oScore: 0 });
  const [gameOver, setGameOver] = useState(false);

  const [sendRequest, setSendRequest] = useState(-1);
  const [data, setData] = useState({ tile: null})
  const [tile, setTile] = useState(null)

    
  const fetchData = async (board) => {
      return await axios.get("https://api.ai-tic-tac-toe.repl.co/predict", {params: {
            board: JSON.stringify(board),
            player: xPlaying ? 1 : 0
          }}
      ) .then((response) => {
          setData(response.data)
      });

  }

  useEffect(() => {
    console.log(sendRequest)
    if (!(sendRequest === -1)) {
      fetchData(board)
    }
  }, [sendRequest]);

  useEffect(() => {
    console.log(data)
    if (!(data.tile === null)) {
      updateBoard(data.tile, !xPlaying)
      setSendRequest(-1)
    }
  }, [data]);

  const updateBoard = (boxIdx, player) => {
    const updatedBoard = board.map((value, idx) => {
      if (idx === boxIdx) {
        return player ? "X" : "O";
      } else {
        return value;
      }
    })

    setBoard(updatedBoard);

    // Check if either player has won the game
    const winner = checkWinner(updatedBoard);

    if (winner) {
      if (winner === "O") {
        let { oScore } = scores;
        oScore += 1;
        setScores({ ...scores, oScore })
      } else {
        let { xScore } = scores;
        xScore += 1;
        setScores({ ...scores, xScore })
      }

    }
  }
  
  const handleBoxClick = (boxIdx) => {
    
    // Step 1: Update the board
    updateBoard(boxIdx, xPlaying)

    // Step 2: Send Request to server
    setSendRequest(true)
  }

  const checkWinner = (board) => {
    for (let i = 0; i < WIN_CONDITIONS.length; i++) {
      const [x, y, z] = WIN_CONDITIONS[i];

      // Iterate through win conditions and check if either player satisfies them
      if (board[x] && board[x] === board[y] && board[y] === board[z]) {
        setGameOver(true);
        return board[x];
      }
    }
  }

  const resetBoard = () => {
    setGameOver(false);
    setBoard(Array(9).fill(null));
  }

  return (
    <div className="App">
      <ScoreBoard scores={scores} xPlaying={xPlaying} />
      <Board board={board} onClick={gameOver ? resetBoard : (boxIdx) => {
        handleBoxClick(boxIdx)
      }} />
      <ResetButton resetBoard={resetBoard} />
    </div>
  );
}

export default App;