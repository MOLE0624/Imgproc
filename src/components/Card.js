import React from "react";
import Image from "../images/image1.jpg";

const Card = ({ children }) => {
  return (
    <>
      <div className="card">
        <div>{children}</div>
        <img src={Image} alt="Image" />
      </div>
    </>
  );
};

export default Card;
