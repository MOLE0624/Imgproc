import React from "react";
import Image from "../images/image1.jpg";

const Card = ({ provided, item, isDragging }) => {
  return (
    <div style={{ paddingBottom: "8px" }}>
      <div
        {...provided.draggableProps}
        {...provided.dragHandleProps}
        ref={provided.innerRef}
        style={{
          ...provided.draggableProps.style,
          opacity: isDragging ? "0.5" : "1",
          color: isDragging ? "green" : "white",
        }}
        className={`item ${isDragging ? "is-dragging" : "card"}`}
      >
        <div>{item.name}</div>
        <div>
          <img src={Image} alt="Image" />
        </div>
      </div>
    </div>
  );
};

export default Card;
