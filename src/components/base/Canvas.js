import React, { useState } from "react";
import CanvasImage from "../../images/image1.jpg";
import Image from "react-image-resizer";
import useMousePosition from "./useMousePosition";
import "../../index.scss";

const Canvas = () => {
  const [canvas_image, setSlippy] = useState(CanvasImage);

  const style = {
    image: {
      // border: "1px solid #ccc",
      background: "#fefefe",
      //   background: "#8f8",
      width: "20%",
      margin: "0 auto",
      // marginTop: 150,
    },
  };

  const position = useMousePosition();
  const x = position.x;
  const y = position.y;

  function mouseDown() {
    console.log(`down x: ${x}, y: ${y}`);
  }

  function mouseUp() {
    console.log(`u  p x: ${x}, y: ${y}`);
  }

  return (
    <>
      <Image
        src={canvas_image}
        onMouseDown={mouseDown}
        onMouseUp={mouseUp}
        alt="Canvas Image"
        className="canvas-image"
        onselectstart="return false;"
        onmousedown="return false;"
        width={900}
        height={450}
        style={style.image}
      />
    </>
  );
};

export default Canvas;
