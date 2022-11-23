import React from "react";
import Image from "../../images/image1.jpg";
import "../../css/Card.scss";

const Card = (props) => {
  return (
    <>
      <div
        className={
          props.name === props.selected.item.name && props.selected.active
            ? "card selected"
            : "card"
        }
      >
        <div>{props.name}</div>
        {/* <div>{selected}</div> */}
        <img src={Image} className="card-image" alt="Image" />
      </div>
    </>
  );
};

export default Card;
