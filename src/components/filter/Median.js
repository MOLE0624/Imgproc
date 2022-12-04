import React, { useEffect, useState } from "react";
import CanvasImage from "../../images/image2.jpg";
import Canvas from "../base/Canvas";
// import Canvas from "../base/Canvas";
// import Controll from "../base/Controll";
// import "../../css/filter.scss";
import "../../css/Median.scss";
import InputSlider from "../slider/InputSlider";

const cv = window.cv;

const Median = () => {
  const [canvasImage, setImage] = useState();
  //   const mat = cv.imread("../../images/image1.jpg");

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

  const bodyScrollLock = require("body-scroll-lock");
  const disableBodyScroll = bodyScrollLock.disableBodyScroll;
  disableBodyScroll("body");
  //   setImage(new cv.Mat());

  const onChangeFile = (e) => {
    if (e.target.files && e.target.files[0]) {
      const img = new Image();
      img.onload = () => {
        const mat = cv.imread(img);
        cv.imshow("canvas-image", mat);
        mat.delete();
      };
      img.src = URL.createObjectURL(e.target.files[0]);
    }
  };

  const onClickInput = () => {
    console.log("hello");
  };

  const onClickGray = () => {
    const mat = cv.imread("canvas-image");
    cv.cvtColor(mat, mat, cv.COLOR_RGBA2GRAY, 0);
    cv.imshow("canvas-image", mat);
    mat.delete();
  };

  const onClickTh = () => {
    const mat = cv.imread("canvas-image");
    if (mat.channels() !== 1) cv.cvtColor(mat, mat, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(mat, mat, 0, 255, cv.THRESH_OTSU);
    cv.imshow("canvas-image", mat);
    mat.delete();
  };

  const onLoad = () => {
    // this.setImage(canvasImage);
    console.log("Loaded!");
  };

  return (
    <>
      <div className="item-canvas">
        <canvas
          id="canvas-image"
          className="canvas-image"
          width={900}
          height={450}
          style={style.image}
        />
        {/* <img src={CanvasImage} alt="canvas-image" id="canvas-image" /> */}
      </div>
      <div>
        {/* <input type="file" onChange={onChangeFile} /> */}
        <button onClick={onClickGray}>グレイスケール化</button>
        <button onClick={onClickTh}>二値化</button>
        {/* <InputSlider onClick={onClickTh} /> */}
      </div>
    </>
  );
};

export default Median;
