import React, { createContext, useState } from "react";
import Board from "./base/Board";
import Canvas from "./base/Canvas";
import Controll from "./base/Controll";
import "../App.scss";
import "../index.scss";
import "../css/grid.scss";

function Main() {
  const [value, setValue] = useState("");

  return (
    <>
      <div className="grid-container">
        <div className="item-board">
          {/* <div className="trello-section"> */}
          <Board name={value} />
        </div>
        <div className="item-canvas">
          <Canvas />
        </div>
        <Controll setValue={setValue} />
      </div>
    </>
  );
}

export default Main;
