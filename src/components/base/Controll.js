import React, { createContext, useContext, useState } from "react";
import Image from "../../images/image1.jpg";
import useMousePosition from "./useMousePosition";
import InputSlider from "../slider/InputSlider";
import { SliderValue } from "../Main";
import "../../index.scss";

const Controll = (props) => {
  const [controll_image, setSlippy] = useState(Image);

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
      <div className="item-controll">
        {/* <img
          src={controll_image}
          onMouseDown={mouseDown}
          onMouseUp={mouseUp}
          alt="Controll Image"
          className="controll-image"
          onselectstart="return false;"
          onmousedown="return false;"
        /> */}
        <div className="controll-slider">
          {/* <InputSlider>{("rho", "0", "200", "1", "30")}</InputSlider> */}
          <InputSlider name="rho" min="0" max="200" step="1" init="50" />
          <p>{props.setValue("test")}</p>
        </div>
      </div>
    </>
  );
};

export default Controll;
